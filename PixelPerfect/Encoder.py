import numpy as np
from bitstring import BitArray, BitStream
from PixelPerfect.Yuv import YuvInfo, YuvBlock, YuvFrame
from PixelPerfect.Coder import Coder, CodecConfig
from PixelPerfect.Decoder import Decoder, IntraFrameDecoder
from math import log2, floor

class Encoder(Coder):
    def __init__(self, video_info: YuvInfo, config: CodecConfig):
        super().__init__(video_info, config)
        self.decoder = Decoder(video_info, config)
        self.bitrate = 0

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
                    
    def RLE_coding(self, data):
        sequence = []
        zero_count = 0
        non_zero_count = 0
        at_last = True;
        for v in reversed(data):
            if v == 0:
                if non_zero_count != 0:
                    sequence.append(non_zero_count)
                    non_zero_count = 0
                zero_count += 1
            else:
                if zero_count != 0:
                    if at_last:
                        sequence.append(0)
                        zero_count = 0
                        at_last = False
                    else:
                        sequence.append(zero_count)
                        zero_count = 0
                non_zero_count -= 1
                at_last = False
                sequence.append(v)
        if non_zero_count != 0:
            sequence.append(non_zero_count)
        else:
            sequence.append(zero_count)
        sequence.reverse()
        return sequence

    def get_diagonal_sequence(self, data):
        max_col = data.shape[1]
        max_row = data.shape[0]
        fdiag = [[] for _ in range(max_row + max_col - 1)]
        for y in range(max_col):
            for x in range(max_row):
                fdiag[x + y].append(data[y, x])
        sequence = []
        for diag in fdiag:
            sequence.extend(diag)
        return sequence

    def entrophy_coding(self, data):
        sequence = self.get_diagonal_sequence(data)
        sequence = self.RLE_coding(sequence)
        bit_sequence = BitStream().join([BitArray(se=i) for i in sequence])
        return bit_sequence

    def cal_entrophy_bitcount(self, data):
        sequence = self.get_diagonal_sequence(data)
        sequence = self.RLE_coding(sequence)
        length = 0
        for v in sequence:
            if v==0:
                length+=1;
            else:
                length += 3 + 2 * floor(log2( abs(v)))
        return length

    def compress_residual(self, residual):
        if self.config.do_approximated_residual:
            residual = self.residual_processor.approx(residual)
        if self.config.do_dct:
            residual = self.residual_processor.dct_transform(residual)
        if self.config.do_quantization:
            residual = self.residual_processor.quantization(residual)
        if self.config.do_entropy:
            residual = self.entrophy_coding(residual)
            self.bitrate += residual.length
        else:
            self.bitrate += self.cal_entrophy_bitcount(residual)
        return residual

    def process(self, frame: YuvFrame):
        compressed_data = []
        self.total_mae = 0
        if self.is_i_frame():
            intra_decoder = IntraFrameDecoder(self.video_info, self.config)
        for block in frame.get_blocks():
            if self.is_p_frame():
                residual, row_mv, col_mv = self.get_inter_data(block)
                residual = self.compress_residual(residual)
                compressed_data.append((residual, row_mv, col_mv))
            else:
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
