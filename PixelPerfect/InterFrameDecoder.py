from PixelPerfect.Yuv import ConstructingFrame, ReferenceFrame
from PixelPerfect.Coder import Coder
from PixelPerfect.CodecConfig import CodecConfig
from typing import Deque
import numpy as np

class InterFrameDecoder(Coder):
    def __init__(self, height, width, previous_frames: Deque[ReferenceFrame], config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.previous_frames = previous_frames
        self.frame = ConstructingFrame(self.config, np.zeros(shape=[height, width], dtype=np.uint8))
        if self.config.need_display:
            self.display_BW_frame = ConstructingFrame(self.config, np.zeros(shape=[height, width], dtype=np.uint8))
            if self.config.DisplayRefFrames:
                self.display_Color_frame = ConstructingFrame(self.config, np.zeros(shape=[height, width], dtype=np.uint8))
            
    # this function should be idempotent
    def process(self, frame_seq, row, col, residual, row_mv, col_mv, is_sub_block: bool):
        ref_frame = self.previous_frames[frame_seq]
        residual = self.decompress_residual(residual, self.config.qp, is_sub_block)
        if is_sub_block:
            block_size = self.config.sub_block_size
        else:
            block_size = self.config.block_size
        reconstructed_block = ref_frame.get_block_by_mv(row, col, row_mv, col_mv, block_size)
        reconstructed_block.add_residual(residual)
        self.frame.put_block(row, col, reconstructed_block)
        if self.config.need_display:
            self.display_BW_frame.put_block(row, col, reconstructed_block)
            if self.config.DisplayMvAndMode:
                self.display_BW_frame.draw_mv(row, col, row_mv, col_mv, block_size)
            if self.config.DisplayRefFrames:
                self.display_Color_frame.draw_ref_frame(row, col, block_size, len(self.previous_frames) - 1 - frame_seq)
            if self.config.DisplayBlocks:
                self.display_BW_frame.draw_block(row, col, block_size)
