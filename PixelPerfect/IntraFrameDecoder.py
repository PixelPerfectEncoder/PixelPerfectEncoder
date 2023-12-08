from PixelPerfect.Yuv import ConstructingFrame
from PixelPerfect.Coder import Coder
from PixelPerfect.CodecConfig import CodecConfig
import numpy as np
class IntraFrameDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame = None
        if self.config.need_display:
            self.display_BW_frame = ConstructingFrame(self.config, np.zeros(shape=[height, width], dtype=np.uint8))
            if self.config.DisplayRefFrames:
                self.display_Color_frame = ConstructingFrame(self.config, np.zeros(shape=[height, width], dtype=np.uint8))

    # this function should be idempotent
    def process(self, row, col, residual, mode, is_sub_block, constructing_frame_data):
        self.frame = ConstructingFrame(self.config, constructing_frame_data)
        residual = self.decompress_residual(residual, self.config.qp, is_sub_block)
        if self.config.ParallelMode == 1:
            ref_block = self.frame.get_plain_ref_block(row, col, is_sub_block)
        else:
            if mode == 0:  # vertical
                ref_block = self.frame.get_vertical_ref_block(row, col, is_sub_block)
            else:  # horizontal
                ref_block = self.frame.get_horizontal_ref_block(row, col, is_sub_block)
        ref_block.add_residual(residual)
        self.frame.put_block(row, col, ref_block)
        
        if self.config.need_display:
            if is_sub_block:
                block_size = self.config.sub_block_size
            else:
                block_size = self.config.block_size
            self.display_BW_frame.put_block(row, col, ref_block)
            if self.config.DisplayBlocks:
                self.display_BW_frame.draw_block(row, col, block_size)
            if self.config.DisplayMvAndMode:
                self.display_BW_frame.draw_mode(row, col, block_size, mode)
