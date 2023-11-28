from PixelPerfect.Yuv import ConstructingFrame, ReferenceFrame
from PixelPerfect.Coder import Coder, VideoCoder
from PixelPerfect.CodecConfig import CodecConfig
from typing import Deque
import numpy as np
import cv2

class IntraFrameDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame = ConstructingFrame(self.config, height=height, width=width)
        if self.config.need_display:
            self.display_BW_frame = ConstructingFrame(self.config, height=height, width=width)
            if self.config.DisplayRefFrames:
                self.display_Color_frame = ConstructingFrame(self.config, height=height, width=width)

    # this function should be idempotent
    def process(self, block_seq, sub_block_seq, residual, mode, is_sub_block, qp):
        row, col = self.get_position_by_seq(block_seq, sub_block_seq)
        residual = self.decompress_residual(residual, qp, is_sub_block)
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

class InterFrameDecoder(Coder):
    def __init__(self, height, width, previous_frames: Deque[ReferenceFrame], config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.previous_frames = previous_frames
        self.frame = ConstructingFrame(self.config, height=height, width=width)
        if self.config.need_display:
            self.display_BW_frame = ConstructingFrame(self.config, height=height, width=width)
            if self.config.DisplayRefFrames:
                self.display_Color_frame = ConstructingFrame(self.config, height=height, width=width)
            
    # this function should be idempotent
    def process(self, frame_seq, block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block: bool, qp: int):
        ref_frame = self.previous_frames[frame_seq]
        row, col = self.get_position_by_seq(block_seq, sub_block_seq)
        residual = self.decompress_residual(residual, qp, is_sub_block)
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

        
class VideoDecoder(VideoCoder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)

    def process_p_frame(self, compressed_data):
        compressed_residual, compressed_descriptors, qp = compressed_data
        descriptors = self.decompress_descriptors(compressed_descriptors)
        inter_decoder = InterFrameDecoder(self.height, self.width, self.previous_frames, self.config)
        block_seq = 0
        sub_block_seq = 0
        last_row_mv, last_col_mv = 0, 0
        total_sub_blocks = 0
        for seq, residual in enumerate(compressed_residual):
            if not self.config.VBSEnable:
                if self.config.FMEEnable:
                    row_mv, col_mv = descriptors[seq * 3] / 2 + last_row_mv, descriptors[seq * 3 + 1] / 2 + last_col_mv
                else:
                    row_mv, col_mv = descriptors[seq * 3] + last_row_mv, descriptors[seq * 3 + 1] + last_col_mv
                frame_seq = descriptors[seq * 3 + 2]
                inter_decoder.process(frame_seq, block_seq, 0, residual, row_mv, col_mv, False, qp)
                last_row_mv, last_col_mv = row_mv, col_mv
                block_seq += 1
            else:
                is_sub_block = descriptors[seq * 4 + 2] == 1
                total_sub_blocks += is_sub_block
                if self.config.FMEEnable:
                    row_mv, col_mv = descriptors[seq * 4] / 2 + last_row_mv, descriptors[seq * 4 + 1] / 2 + last_col_mv
                else:
                    row_mv, col_mv = descriptors[seq * 4] + last_row_mv, descriptors[seq * 4 + 1] + last_col_mv
                frame_seq = descriptors[seq * 4 + 3]
                inter_decoder.process(frame_seq, block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block, qp)
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
        if self.config.need_display:
            if self.config.DisplayRefFrames:
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                img[:, :, 0] = np.full((self.height, self.width), 128, dtype=np.uint8)
                img[:, :, 1] = inter_decoder.display_Color_frame.data
                img[:, :, 2] = inter_decoder.display_BW_frame.data
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            else:
                img = inter_decoder.display_BW_frame.data
            cv2.imshow("", img)
            cv2.waitKey(1)
        self.sub_block_ratio = (total_sub_blocks / 4) / len(compressed_residual)
        return frame

    def process_i_frame(self, compressed_data):
        compressed_residual, compressed_descriptors, qp = compressed_data
        descriptors = self.decompress_descriptors(compressed_descriptors)
        intra_decoder = IntraFrameDecoder(self.height, self.width, self.config)
        block_seq = 0
        sub_block_seq = 0
        for seq, residual in enumerate(compressed_residual):
            if not self.config.VBSEnable:
                intra_decoder.process(block_seq, 0, residual, descriptors[seq], False, qp)
                block_seq += 1
            else:
                is_sub_block = descriptors[seq * 2 + 1] == 1
                intra_decoder.process(block_seq, sub_block_seq, residual, descriptors[seq * 2], is_sub_block, qp)
                if is_sub_block:
                    sub_block_seq += 1
                    if sub_block_seq == 4:
                        sub_block_seq = 0
                        block_seq += 1
                else:
                    block_seq += 1
                
        frame = intra_decoder.frame.to_reference_frame()
        self.frame_processed(frame)
        if self.config.need_display:      
            cv2.imshow("", intra_decoder.display_BW_frame.data)
            cv2.waitKey(1)
        return frame

    def process(self, compressed_data):
        if self.is_p_frame():
            return self.process_p_frame(compressed_data)
        else:
            return self.process_i_frame(compressed_data)
