from PixelPerfect.Yuv import YuvBlock, ReferenceFrame
from PixelPerfect.Coder import Coder
from PixelPerfect.InterFrameDecoder import InterFrameDecoder
from PixelPerfect.CodecConfig import CodecConfig
from typing import Deque

class InterFrameEncoder(Coder):
    def __init__(self, height, width, previous_frames: Deque[ReferenceFrame], config: CodecConfig):
        super().__init__(height, width, config)
        self.previous_frames = previous_frames
        self.inter_decoder = InterFrameDecoder(height, width, config)

    def is_better_match_block(
        self, block: YuvBlock, ref_block: YuvBlock, min_mae, best_i, best_j
    ) -> bool:
        mae = block.get_mae(ref_block)
        if mae > min_mae:
            return False, None
        if mae < min_mae:
            return True, mae
        di = block.row - ref_block.row
        dj = block.col - ref_block.col
        # if mae is equal, we need to compare the distance
        if abs(di) + abs(dj) > abs(best_i - block.row) + abs(best_j - block.col):
            return False, None
        if abs(di) + abs(dj) < abs(best_i - block.row) + abs(best_j - block.col):
            return True, mae
        # if distance is equal, we prefer the smaller di
        if di > best_i - block.row:
            return False, None
        if di < best_i - block.row:
            return True, mae
        # if di is equal, we prefer the smaller dj
        if dj < best_j - block.col:
            return True, mae
        return False, None

    def get_inter_data_fast_search(self, block: YuvBlock, mv_row_pred, mv_col_pred):
        best_block_among_all_frames = None
        best_mae_among_all_frames = float("inf")
        best_ref_frame_seq = None
        for ref_frame_seq, ref_frame in enumerate(self.previous_frames):
            best_block = ref_frame.get_block(
                min(max(0, block.row + mv_row_pred), self.height - block.block_size),
                min(max(0, block.col + mv_col_pred), self.width - block.block_size),
                block.block_size == self.config.sub_block_size,
            )
            best_mae = block.get_mae(best_block)
            has_gain = True
            within_limit = True
            while has_gain and within_limit:
                has_gain = False
                # do cross area search
                for ref_block in ref_frame.get_ref_blocks_in_cross_area(best_block):
                    ref_block_mae = block.get_mae(ref_block)
                    if ref_block_mae < best_mae:
                        best_block = ref_block
                        best_mae = ref_block_mae
                        has_gain = True
                if has_gain and max(abs(best_block.row - block.row), abs(best_block.col - block.col)) >= self.config.FastME_LIMIT:
                    within_limit = False
            if best_mae < best_mae_among_all_frames:
                best_block_among_all_frames = best_block
                best_mae_among_all_frames = best_mae
                best_ref_frame_seq = ref_frame_seq
            
        return (
            block.get_residual(best_block_among_all_frames),
            best_block_among_all_frames.row - block.row,
            best_block_among_all_frames.col - block.col,
            best_ref_frame_seq
        )

    def get_inter_data_normal_search(self, block: YuvBlock) :
        min_mae = float("inf")
        best_i, best_j = None, None
        best_di, best_dj = None, None
        best_block = None
        best_frame_seq = None
        for frame_seq, ref_frame in enumerate(self.previous_frames):
            for ref_block in ref_frame.get_ref_blocks_in_offset_area(block):
                is_better_match, mae = self.is_better_match_block(block, ref_block, min_mae, best_i, best_j)
                if is_better_match:
                    min_mae = mae
                    best_i, best_j = ref_block.row, ref_block.col
                    best_di, best_dj = ref_block.row - block.row, ref_block.col - block.col
                    best_block = ref_block
                    best_frame_seq = frame_seq
        return block.get_residual(best_block), best_di, best_dj, best_frame_seq

    def get_inter_data(self, block: YuvBlock, last_row_mv, last_col_mv):
        if self.config.FastME:
            return self.get_inter_data_fast_search(block, last_row_mv, last_col_mv)
        else:
            return self.get_inter_data_normal_search(block)

    # this function should be idempotent
    def process(self, block: YuvBlock, last_row_mv: int, last_col_mv: int, use_sub_blocks: bool, constructing_frame_data):
        compressed_residual = []
        descriptors = []
        residual_bitrate = 0
        if use_sub_blocks:
            for sub_block in block.get_sub_blocks():
                if self.config.ParallelMode == 1:
                    last_row_mv, last_col_mv = 0, 0
                residual, row_mv, col_mv, frame_seq = self.get_inter_data(sub_block, last_row_mv, last_col_mv)
                residual, bitrate = self.compress_residual(residual, self.config.qp, is_sub_block=True)
                compressed_residual.append(residual)
                residual_bitrate += bitrate
                if self.config.FMEEnable:
                    descriptors.append(int(2 * (row_mv - last_row_mv)))
                    descriptors.append(int(2 * (col_mv - last_col_mv)))
                else:
                    descriptors.append(row_mv - last_row_mv)
                    descriptors.append(col_mv - last_col_mv)
                descriptors.append(1)
                descriptors.append(frame_seq)
                last_row_mv, last_col_mv = row_mv, col_mv
                self.inter_decoder.process(
                    ref_frame=self.previous_frames[frame_seq], 
                    row=sub_block.row, 
                    col=sub_block.col,
                    residual=residual, 
                    row_mv=row_mv, 
                    col_mv=col_mv, 
                    is_sub_block=True,
                    constructing_frame_data=constructing_frame_data,
                )
        else:
            residual, row_mv, col_mv, frame_seq = self.get_inter_data(block, last_row_mv, last_col_mv)
            residual, bitrate = self.compress_residual(residual, self.config.qp, is_sub_block=False)
            residual_bitrate += bitrate
            compressed_residual.append(residual)
            if self.config.FMEEnable:
                descriptors.append(int(2 * (row_mv - last_row_mv)))
                descriptors.append(int(2 * (col_mv - last_col_mv)))
            else:
                descriptors.append(row_mv - last_row_mv)
                descriptors.append(col_mv - last_col_mv)
            if self.config.VBSEnable:
                descriptors.append(0)
            descriptors.append(frame_seq)
            last_row_mv, last_col_mv = row_mv, col_mv
            self.inter_decoder.process(
                ref_frame=self.previous_frames[frame_seq], 
                row=block.row,
                col=block.col, 
                residual=residual, 
                row_mv=row_mv, 
                col_mv=col_mv, 
                is_sub_block=False,
                constructing_frame_data=constructing_frame_data,
            )
        reconstructed_block = self.inter_decoder.frame.get_block(block.row, block.col, is_sub_block=False)
        distortion = block.get_SAD(reconstructed_block)
        return (
            compressed_residual,
            descriptors,
            distortion,
            residual_bitrate,
            last_row_mv,
            last_col_mv,
        )