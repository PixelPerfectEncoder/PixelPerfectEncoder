from PixelPerfect.Yuv import YuvFrame
from PixelPerfect.Coder import CodecConfig, Coder, VideoCoder
import numpy as np
import math


class IntraFrameDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame = np.zeros([self.padded_height, self.padded_width], dtype=np.uint8)

    # this function should be idempotent
    def process(self, block_seq, sub_block_seq, residual, mode, is_sub_block):
        row, col = self.get_position_by_seq(block_seq, sub_block_seq)
        if is_sub_block:
            block_size = self.config.sub_block_size
        else:
            block_size = self.config.block_size
        residual = self.decompress_residual(residual, block_size)
        ref_block = np.full([block_size, block_size], 128)
        if mode == 0:  # vertical
            if row != 0:
                ref_row = self.frame[row - 1 : row, col : col + block_size]
                ref_block = np.repeat(ref_row, repeats=block_size, axis=0)
        else:  # horizontal
            if col != 0:
                ref_col = self.frame[row : row + block_size, col - 1 : col]
                ref_block = np.repeat(ref_col, repeats=block_size, axis=1)
        self.frame[row : row + block_size, col : col + block_size] = residual + ref_block

class InterFrameDecoder(Coder):
    def __init__(self, height, width, previous_frame, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.previous_frame = previous_frame
        self.frame = np.zeros([self.padded_height, self.padded_width], dtype=np.uint8)
        
    # this function should be idempotent
    def process(self, block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block):        
        row, col = self.get_position_by_seq(block_seq, sub_block_seq)
        if is_sub_block:
            block_size = self.config.sub_block_size
        else:
            block_size = self.config.block_size
        residual = self.decompress_residual(residual, block_size)
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
            self.frame[
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
            self.frame[
                row : row + block_size, col : col + block_size
            ] = reconstructed_block
        
class VideoDecoder(VideoCoder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)

    def process_p_frame(self, compressed_data):
        compressed_residual, compressed_descriptors = compressed_data
        descriptors = self.decompress_descriptors(compressed_descriptors)
        inter_decoder = InterFrameDecoder(self.height, self.width, self.previous_frame, self.config)
        block_seq = 0
        sub_block_seq = 0
        last_row_mv, last_col_mv = 0, 0
        for seq, residual in enumerate(compressed_residual):
            if not self.config.VBSEnable:
                row_mv, col_mv = descriptors[seq * 2] + last_row_mv, descriptors[seq * 2 + 1] + last_col_mv
                inter_decoder.process(block_seq, 0, residual, row_mv, col_mv, False)
                last_row_mv, last_col_mv = row_mv, col_mv
                block_seq += 1
            else:
                is_sub_block = descriptors[seq * 3 + 2] == 1
                row_mv, col_mv = descriptors[seq * 3] + last_row_mv, descriptors[seq * 3 + 1] + last_col_mv
                inter_decoder.process(block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block)
                last_row_mv, last_col_mv = row_mv, col_mv
                if is_sub_block:
                    sub_block_seq += 1
                    if sub_block_seq == 4:
                        sub_block_seq = 0
                        block_seq += 1
                else:
                    block_seq += 1
        frame = np.clip(inter_decoder.frame, 0, 255)
        res = YuvFrame(frame, self.config.block_size)
        self.frame_processed(res)
        return res

    def process_i_frame(self, compressed_data):
        compressed_residual, compressed_descriptors = compressed_data
        descriptors = self.decompress_descriptors(compressed_descriptors)
        intra_decoder = IntraFrameDecoder(self.height, self.width, self.config)
        block_seq = 0
        sub_block_seq = 0
        for seq, residual in enumerate(compressed_residual):
            if not self.config.VBSEnable:
                intra_decoder.process(block_seq, 0, residual, descriptors[seq], False)
                block_seq += 1
            else:
                is_sub_block = descriptors[seq * 2 + 1] == 1
                intra_decoder.process(block_seq, sub_block_seq, residual, descriptors[seq * 2], is_sub_block)
                if is_sub_block:
                    sub_block_seq += 1
                    if sub_block_seq == 4:
                        sub_block_seq = 0
                        block_seq += 1
                else:
                    block_seq += 1
                
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
