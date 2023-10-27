import numpy as np
from bitstring import BitArray, BitStream
from PixelPerfect.Yuv import YuvInfo, YuvBlock
from PixelPerfect.Coder import Coder, CodecConfig
from PixelPerfect.Decoder import Decoder
from PixelPerfect.FileIO import read_frames


class Encoder(Coder):
    def __init__(self, video_info: YuvInfo, config: CodecConfig, source_path: str):
        super().__init__(video_info, config)
        self.decoder = Decoder(video_info, config)
        self.source_path = source_path

    def is_better_match_block(
        self, di, dj, block: YuvBlock, min_mae, best_i, best_j
    ) -> bool:
        i = block.row_position + di
        j = block.col_position + dj
        block_size = self.config.block_size
        if (
            0 <= i <= self.video_info.height - block_size
            and 0 <= j <= self.video_info.width - block_size
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

    def find_best_match_block(self, block: YuvBlock) -> YuvBlock:
        min_mae = float("inf")
        best_i, best_j = None, None
        offset = self.config.block_search_offset
        for di in range(-offset, offset + 1):
            for dj in range(-offset, offset + 1):
                is_better_match, mae = self.is_better_match_block(
                    di, dj, block, min_mae, best_i, best_j
                )
                if is_better_match:
                    min_mae = mae
                    best_i, best_j = block.row_position + di, block.col_position + dj

        block_size = self.config.block_size
        return YuvBlock(
            self.previous_frame.data[
                best_i : best_i + block_size, best_j : best_j + block_size
            ],
            block_size,
            best_i,
            best_j,
        )

    def intra_horizontal_pred(self, cur_block):
        if cur_block.col_position == 0:
            predicted_block = np.full((128, 128), cur_block.block_size)
        else:
            ref_col = self.cur_frame[
                cur_block.row_position : cur_block.row_position + cur_block.block_size,
                cur_block.col_position - 1 : cur_block.col_position,
            ]
            predicted_block = ref_col
            while predicted_block.shape[1] < cur_block.block_size:
                np.column_stack((predicted_block, ref_col))
        residual = np.abs(predicted_block - cur_block.data)
        MAE = np.sum(residual)
        return (MAE, residual, predicted_block)

    def intra_vertical_pred(self, cur_block):
        if cur_block.row_position == 0:
            predicted_block = np.full((128, 128), cur_block.block_size)
        else:
            ref_col = self.cur_frame[
                cur_block.row_position - 1 : cur_block.row_position,
                cur_block.col_position : cur_block.col_position + cur_block.block_size,
            ]
            predicted_block = np.tile(ref_col, (cur_block.block_size, 1))
        residual = np.abs(predicted_block - cur_block.data)
        MAE = np.sum(residual)
        return (MAE, residual, predicted_block)

    def intra_pred(self, cur_block):
        MAE_h, residual_h, predicted_block_h = self.intra_horizontal_pred(cur_block)
        MAE_v, residual_v, predicted_block_v = self.intra_vertical_pred(cur_block)
        # return mode 0 if horizontal mode has leat MAE
        if MAE_h < MAE_v:
            reconstructed_block = residual_h + predicted_block_h
            return (0, residual_h, reconstructed_block)
        else:
            reconstructed_block = residual_v + predicted_block_v
            return (1, residual_v, reconstructed_block)

    def RLE_coding(self, data):
        sequence = []
        zero_count = 0
        non_zero_count = 0
        for v in reversed(data):
            if v == 0:
                if non_zero_count != 0:
                    sequence.append(non_zero_count)
                    non_zero_count = 0
                zero_count += 1
            else:
                if zero_count != 0:
                    sequence.append(zero_count)
                    zero_count = 0
                non_zero_count -= 1
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
        sequence = [len(sequence)] + sequence
        bit_sequence = BitStream().join([BitArray(se=i) for i in sequence])
        return bit_sequence

    def process(self):
        for frame in read_frames(self.source_path, self.video_info):
            compressed_data = []
            if self.is_p_frame():
                for block in frame.get_blocks(self.config.block_size):
                    best_match_block = self.find_best_match_block(block)
                    residual = block.get_residual(best_match_block.data)
                    if self.config.do_approximated_residual:
                        residual = self.residual_processor.approx(residual)
                    if self.config.do_dct:
                        residual = self.residual_processor.dct_transform(residual)
                    if self.config.do_quantization:
                        residual = self.residual_processor.quantization(residual)
                    if self.config.do_entropy:
                        residual = self.entrophy_coding(residual)
                    compressed_data.append(
                        (
                            best_match_block.row_position,
                            best_match_block.col_position,
                            residual,
                        )
                    )
            else:
                pass

            yield compressed_data
            self.frame_processed(self.decoder.process(compressed_data))
