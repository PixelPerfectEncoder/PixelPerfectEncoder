from PixelPerfect.InterFrameDecoder import InterFrameDecoder
from PixelPerfect.IntraFrameDecoder import IntraFrameDecoder
from PixelPerfect.Yuv import ReferenceFrame
from PixelPerfect.Coder import Coder
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.CompressedFrameData import CompressedFrameData
from PixelPerfect.BlockDescriptors import PBlockDescriptors, IBlockDescriptors
from typing import List
import numpy as np
import cv2

class VideoDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)

    def get_position_by_seq(self, block_seq, sub_block_seq):
        row = block_seq // self.row_block_num * self.config.block_size
        col = block_seq % self.row_block_num * self.config.block_size
        row += (sub_block_seq // 2) * self.config.sub_block_size
        col += (sub_block_seq % 2) * self.config.sub_block_size
        return row, col
    
    def deserialize_PFrame_descriptors(self, serialize_descriptors) -> List[PBlockDescriptors]:
        seq = 0
        descriptors = []
        if self.config.FMEEnable:
            mv_divider = 2
        else:
            mv_divider = 1
        if self.config.ParallelMode == 1:
            last_mv_muiplier = 0
        else:
            last_mv_muiplier = 1
        if not self.config.VBSEnable:
            block_seq = 0
            last_row_mv, last_col_mv = 0, 0
            while seq < len(serialize_descriptors):
                row, col = self.get_position_by_seq(block_seq, 0)
                if col == 0:
                    last_row_mv, last_col_mv = 0, 0
                des = PBlockDescriptors(
                    row_mv = serialize_descriptors[seq] / mv_divider + last_row_mv * last_mv_muiplier,
                    col_mv = serialize_descriptors[seq + 1] / mv_divider + last_col_mv * last_mv_muiplier,
                    is_sub_block = False,
                    frame_seq = serialize_descriptors[seq + 2],
                    row = row,
                    col = col,
                )
                seq += 3
                last_row_mv, last_col_mv = des.row_mv, des.col_mv
                descriptors.append(des)
                block_seq += 1
        else:
            last_row_mv, last_col_mv = 0, 0
            block_seq = 0
            sub_block_seq = 0
            while seq < len(serialize_descriptors):
                row, col = self.get_position_by_seq(block_seq, sub_block_seq)
                if col == 0 and sub_block_seq == 0:
                    last_row_mv, last_col_mv = 0, 0
                des = PBlockDescriptors(
                    row_mv = serialize_descriptors[seq] / mv_divider + last_row_mv * last_mv_muiplier,
                    col_mv = serialize_descriptors[seq + 1] / mv_divider + last_col_mv * last_mv_muiplier,
                    is_sub_block = serialize_descriptors[seq + 2] == 1,
                    frame_seq = serialize_descriptors[seq + 3],
                    row = row,
                    col = col,
                )
                seq += 4
                last_row_mv, last_col_mv = des.row_mv, des.col_mv
                descriptors.append(des)
                if des.is_sub_block:
                    sub_block_seq += 1
                    if sub_block_seq == 4:
                        sub_block_seq = 0
                        block_seq += 1
                else:
                    block_seq += 1
        return descriptors
                
    def deserialize_IFrame_descriptors(self, serialize_descriptors) -> List[IBlockDescriptors]:
        seq = 0
        descriptors = []
        if not self.config.VBSEnable:
            while seq < len(serialize_descriptors):
                row, col = self.get_position_by_seq(seq, 0)
                des = IBlockDescriptors(
                    mode = serialize_descriptors[seq],
                    is_sub_block = False,
                    row = row,
                    col = col,
                )
                seq += 1
                descriptors.append(des)
        else:
            block_seq = 0
            sub_block_seq = 0
            while seq < len(serialize_descriptors):
                row, col = self.get_position_by_seq(block_seq, sub_block_seq)
                des = IBlockDescriptors(
                    mode = serialize_descriptors[seq],
                    is_sub_block = serialize_descriptors[seq + 1] == 1,
                    row = row,
                    col = col,
                )
                seq += 2
                descriptors.append(des)
                if des.is_sub_block:
                    sub_block_seq += 1
                    if sub_block_seq == 4:
                        sub_block_seq = 0
                        block_seq += 1
                else:
                    block_seq += 1
        return descriptors
            
    def process_p_frame(self, compressed_data: CompressedFrameData, previous_frames: List[ReferenceFrame]):
        compressed_residual = compressed_data.residual
        compressed_descriptors = compressed_data.descriptors
        serialize_descriptors = self.decompress_descriptors(compressed_descriptors)
        descriptors = self.deserialize_PFrame_descriptors(serialize_descriptors)
        inter_decoder = InterFrameDecoder(self.height, self.width, self.config)
        total_sub_blocks = 0
        constructing_frame_data = np.zeros(shape=[self.height, self.width], dtype=np.uint8)
        for residual, des in zip(compressed_residual, descriptors):
            inter_decoder.process(
                ref_frame=previous_frames[des.frame_seq], 
                row=des.row, 
                col=des.col, 
                residual=residual, 
                row_mv=des.row_mv, 
                col_mv=des.col_mv, 
                is_sub_block=des.is_sub_block,
                constructing_frame_data=constructing_frame_data,
            )
            total_sub_blocks += des.is_sub_block

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
        
        frame = inter_decoder.frame.to_reference_frame()
        return frame

    def process_i_frame(self, compressed_data: CompressedFrameData):
        compressed_residual = compressed_data.residual
        compressed_descriptors = compressed_data.descriptors
        constructing_frame_data = np.zeros(shape=[self.height, self.width], dtype=np.uint8)
        serialize_descriptors = self.decompress_descriptors(compressed_descriptors)
        descriptors = self.deserialize_IFrame_descriptors(serialize_descriptors)
        intra_decoder = IntraFrameDecoder(self.height, self.width, self.config)
        for residual, des in zip(compressed_residual, descriptors):
            intra_decoder.process(
                row=des.row, 
                col=des.col, 
                residual=residual, 
                mode=des.mode, 
                is_sub_block=des.is_sub_block, 
                constructing_frame_data=constructing_frame_data
            )
        if self.config.need_display:
            cv2.imshow("", intra_decoder.display_BW_frame.data)
            cv2.waitKey(1)
        frame = intra_decoder.frame.to_reference_frame()
        return frame

    def process(self, compressed_data, is_i_frame: bool, previous_frames: List[ReferenceFrame]):
        if is_i_frame:
            return self.process_i_frame(compressed_data)
        else:
            return self.process_p_frame(compressed_data, previous_frames)
