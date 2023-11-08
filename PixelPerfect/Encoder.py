import numpy as np
from PixelPerfect.Yuv import YuvBlock, YuvFrame
from PixelPerfect.Coder import Coder, CodecConfig
from PixelPerfect.Decoder import Decoder, IntraFrameDecoder


class Encoder(Coder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)
        self.decoder = Decoder(height, width, config)

    def is_better_match_block(
        self, di, dj, block: YuvBlock, min_mae, best_i, best_j
    ) -> bool:
        i = block.row_position + di
        j = block.col_position + dj
        block_size = block.block_size
        if (
            0 <= i <= self.previous_frame.height - block_size
            and 0 <= j <= self.previous_frame.width - block_size
        ):
            reference_block_data = self.previous_frame.data[
                i : i + block_size, j : j + block_size
            ]
            mae = block.get_mae(reference_block_data)
            if mae > min_mae:
                return False, None
            if mae < min_mae:
                return True, mae
            if abs(di) + abs(dj) > abs(best_i - block.row_position) + abs(
                best_j - block.col_position
            ):
                return False, None
            if abs(di) + abs(dj) < abs(best_i - block.row_position) + abs(
                best_j - block.col_position
            ):
                return True, mae
            if di > best_i - block.row_position:
                return False, None
            if di < best_i - block.row_position:
                return True, mae
            if dj < best_j - block.col_position:
                return True, mae
        return False, None

    def get_inter_data(self, block: YuvBlock):
        min_mae = float("inf")
        best_i, best_j = None, None
        best_di, best_dj = None, None
        offset = self.config.block_search_offset
        for di in range(-offset, offset + 1):
            for dj in range(-offset, offset + 1):
                is_better_match, mae = self.is_better_match_block(
                    di, dj, block, min_mae, best_i, best_j
                )
                if is_better_match:
                    min_mae = mae
                    best_i, best_j = block.row_position + di, block.col_position + dj
                    best_di, best_dj = di, dj
        block_size = self.config.block_size
        self.total_mae += min_mae
        return block.get_residual(self.previous_frame.data[best_i : best_i + block_size, best_j : best_j + block_size]), best_di, best_dj

    def get_intra_data(self, vertical_ref, horizontal_ref, block: YuvBlock):
        vertical_residual = block.data.astype(np.int16) - vertical_ref.astype(np.int16)  
        horizontal_residual = block.data.astype(np.int16) - horizontal_ref.astype(np.int16)
        vertical_mae = np.mean(np.abs(vertical_residual))
        horizontal_mae = np.mean(np.abs(horizontal_residual))
        if vertical_mae < horizontal_mae:
            return vertical_residual, 0
        else:
            return horizontal_residual, 1
                    
    def process_p_frame(self, frame: YuvFrame):
        compressed_residual = []
        descriptors = []
        self.total_mae = 0
        last_row_mv, last_col_mv = 0, 0
        for block in frame.get_blocks():
            residual, row_mv, col_mv = self.get_inter_data(block)
            residual = self.compress_residual(residual)
            compressed_residual.append(residual)
            descriptors.append(row_mv - last_row_mv)
            descriptors.append(col_mv - last_col_mv)
            last_row_mv, last_col_mv = row_mv, col_mv
            
        compressed_data = (compressed_residual, self.compress_descriptors(descriptors))
        decoded_frame = self.decoder.process(compressed_data)
        self.frame_processed(decoded_frame)
        self.average_mae = self.total_mae / len(compressed_data)
        return compressed_data
    
    def process_i_frame(self, frame: YuvFrame):
        compressed_data = []
        self.total_mae = 0
        intra_decoder = IntraFrameDecoder(self.height, self.width, self.config)
        for block in frame.get_blocks():
            vertical_ref = np.full([block.block_size, block.block_size], 128)
            if block.row_position != 0:
                vertical_ref_row = intra_decoder.frame[
                    block.row_position - 1 : block.row_position,
                    block.col_position : block.col_position + block.block_size,
                ]
                vertical_ref = np.repeat(vertical_ref_row, repeats=block.block_size, axis=0)
                
            horizontal_ref = np.full([block.block_size, block.block_size], 128)
            if block.col_position != 0:
                horizontal_ref_col = intra_decoder.frame[
                    block.row_position : block.row_position + block.block_size,
                    block.col_position - 1 : block.col_position,
                ]
                horizontal_ref = np.repeat(horizontal_ref_col, repeats=block.block_size, axis=1)
                
            residual, mode = self.get_intra_data(vertical_ref, horizontal_ref, block)
            residual = self.compress_residual(residual)
            intra_decoder.process((residual, mode))
            compressed_data.append((residual, mode))
        decoded_frame = self.decoder.process(compressed_data)
        self.frame_processed(decoded_frame)
        self.average_mae = self.total_mae / len(compressed_data)
        return compressed_data

    def process(self, frame: YuvFrame):
        if self.is_i_frame():
            return self.process_i_frame(frame)
        else:
            return self.process_p_frame(frame)