from PixelPerfect.Yuv import YuvFrame
from PixelPerfect.Coder import CodecConfig, Coder
import numpy as np
import math


class IntraFrameDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame = np.zeros(
            [self.previous_frame.height, self.previous_frame.width], dtype=np.uint8
        )
        self.seq = 0

    def process(self, residual, mode):
        block_size = self.config.block_size
        row_block_num = self.previous_frame.width // block_size
        row = self.seq // row_block_num * block_size
        col = self.seq % row_block_num * block_size
        residual = self.decompress_residual(residual)
        ref_block = np.full([block_size, block_size], 128)
        if mode == 0:  # vertical
            if row != 0:
                ref_row = self.frame[row - 1 : row, col : col + block_size]
                ref_block = np.repeat(ref_row, repeats=block_size, axis=0)
        else:  # horizontal
            if col != 0:
                ref_col = self.frame[row : row + block_size, col - 1 : col]
                ref_block = np.repeat(ref_col, repeats=block_size, axis=1)
        self.frame[row : row + block_size, col : col + block_size] = (
            residual + ref_block
        )
        self.seq += 1


class Decoder(Coder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)

    def process_p_frame(self, compressed_data):
        compressed_residual, compressed_descriptors = compressed_data
        descriptors = self.decompress_descriptors(compressed_descriptors)
        frame = np.zeros(
            [self.previous_frame.height, self.previous_frame.width], dtype=np.uint8
        )
        block_size = self.config.block_size
        row_block_num = self.previous_frame.width // block_size
        last_row_mv, last_col_mv = 0, 0
        for seq, residual in enumerate(compressed_residual):
            row = seq // row_block_num * block_size
            col = seq % row_block_num * block_size
            diff_row_mv, diff_col_mv = descriptors[seq * 2], descriptors[seq * 2 + 1]
            row_mv, col_mv = diff_row_mv + last_row_mv, diff_col_mv + last_col_mv
            last_row_mv, last_col_mv = row_mv, col_mv
            residual = self.decompress_residual(residual)
            if self.config.FMEEnable:
                ref_row_f = math.floor(row + row_mv / 2)
                ref_col_f = math.floor(col + col_mv / 2)
                ref_row_c = math.ceil(row + row_mv / 2)
                ref_col_c = math.ceil(col + col_mv / 2)
                ref_blcok = np.round(
                    (
                        self.previous_frame.data[
                            ref_row_f : ref_row_f + block_size,
                            ref_col_f : ref_col_f + block_size,
                        ]
                        / 2
                        + self.previous_frame.data[
                            ref_row_c : ref_row_c + block_size,
                            ref_col_c : ref_col_c + block_size,
                        ]
                        / 2
                    )
                )
                reconstructed_block = ref_blcok + residual
                frame[
                    row : row + block_size, col : col + block_size
                ] = reconstructed_block

            else:
                ref_row = row + row_mv
                ref_col = col + col_mv
                reconstructed_block = (
                    self.previous_frame.data[
                        ref_row : ref_row + block_size, ref_col : ref_col + block_size
                    ]
                    + residual
                )
                frame[
                    row : row + block_size, col : col + block_size
                ] = reconstructed_block
        frame = np.clip(frame, 0, 255)
        res = YuvFrame(frame, self.config.block_size)
        self.frame_processed(res)
        return res

    def process_i_frame(self, compressed_data):
        compressed_residual, compressed_descriptors = compressed_data
        descriptors = self.decompress_descriptors(compressed_descriptors)
        intra_decoder = IntraFrameDecoder(self.height, self.width, self.config)
        for seq, residual in enumerate(compressed_residual):
            intra_decoder.process(residual, descriptors[seq])
        frame = intra_decoder.frame
        frame = np.clip(frame, 0, 255)
        res = YuvFrame(frame, self.config.block_size)
        self.frame_processed(res)
        return res

    def process(self, compressed_data):
        if self.is_p_frame():
            return self.process_p_frame(compressed_data)
        else:
            return self.process_i_frame(compressed_data)
