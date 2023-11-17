from PixelPerfect.Yuv import ConstructingFrame, ReferenceFrame
from PixelPerfect.Coder import Coder, VideoCoder
from PixelPerfect.CodecConfig import CodecConfig
import numpy as np
import math


class IntraFrameDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame = ConstructingFrame(self.config, height=height, width=width)

    # this function should be idempotent
    def process(self, block_seq, sub_block_seq, residual, mode, is_sub_block):
        row, col = self.get_position_by_seq(block_seq, sub_block_seq)
        if is_sub_block:
            block_size = self.config.sub_block_size
        else:
            block_size = self.config.block_size
        residual = self.decompress_residual(residual, block_size)
        ref_block = np.full([block_size, block_size], 128, dtype=np.uint8)
        if mode == 0:  # vertical
            ref_block = self.frame.get_vertical_ref_block(row, col, is_sub_block)
        else:  # horizontal
            ref_block = self.frame.get_horizontal_ref_block(row, col, is_sub_block)
        ref_block.add_residual(residual)
        self.frame.put_block(row, col, ref_block)

class InterFrameDecoder(Coder):
    def __init__(self, height, width, previous_frame: ReferenceFrame, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.previous_frame = previous_frame
        self.frame = ConstructingFrame(self.config, height=height, width=width)
        
    # this function should be idempotent
    def process(self, block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block):        
        row, col = self.get_position_by_seq(block_seq, sub_block_seq)
        if is_sub_block:
            block_size = self.config.sub_block_size
        else:
            block_size = self.config.block_size
        residual = self.decompress_residual(residual, block_size)
        reconstructed_block = self.previous_frame.get_block_by_mv(row, col, row_mv, col_mv, block_size)
        reconstructed_block.add_residual(residual)
        self.frame.put_block(row, col, reconstructed_block)
        
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
                if self.config.FMEEnable:
                    row_mv, col_mv = descriptors[seq * 2] / 2 + last_row_mv, descriptors[seq * 2 + 1] / 2 + last_col_mv
                else:
                    row_mv, col_mv = descriptors[seq * 2] + last_row_mv, descriptors[seq * 2 + 1] + last_col_mv
                inter_decoder.process(block_seq, 0, residual, row_mv, col_mv, False)
                last_row_mv, last_col_mv = row_mv, col_mv
                block_seq += 1
            else:
                is_sub_block = descriptors[seq * 3 + 2] == 1
                if self.config.FMEEnable:
                    row_mv, col_mv = descriptors[seq * 3] / 2 + last_row_mv, descriptors[seq * 3 + 1] / 2 + last_col_mv
                else:
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
        frame = inter_decoder.frame.to_reference_frame()
        self.frame_processed(frame)
        return frame

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
                
        frame = intra_decoder.frame.to_reference_frame()
        self.frame_processed(frame)
        return frame

    def process(self, compressed_data):
        if self.is_p_frame():
            return self.process_p_frame(compressed_data)
        else:
            return self.process_i_frame(compressed_data)
