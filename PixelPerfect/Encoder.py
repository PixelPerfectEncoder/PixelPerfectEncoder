from PixelPerfect.Yuv import YuvBlock, ReferenceFrame
from PixelPerfect.Coder import Coder, VideoCoder
from PixelPerfect.Decoder import IntraFrameDecoder, InterFrameDecoder
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.BitRateController import BitRateController
from typing import Deque
import copy
class InterFrameEncoder(Coder):
    def __init__(self, height, width, previous_frames: Deque[ReferenceFrame], config: CodecConfig):
        super().__init__(height, width, config)
        self.previous_frames = previous_frames
        self.inter_decoder = InterFrameDecoder(height, width, previous_frames, config)

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
    def process(self, block: YuvBlock, block_seq: int, last_row_mv: int, last_col_mv: int, use_sub_blocks: bool, qp: int):
        compressed_residual = []
        descriptors = []
        residual_bitrate = 0
        if use_sub_blocks:
            for sub_block_seq, sub_block in enumerate(block.get_sub_blocks()):
                residual, row_mv, col_mv, frame_seq = self.get_inter_data(sub_block, last_row_mv, last_col_mv)
                residual, bitrate = self.compress_residual(residual, qp=qp, is_sub_block=True)
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
                self.inter_decoder.process(frame_seq, block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block=True, qp=qp)
        else:
            residual, row_mv, col_mv, frame_seq = self.get_inter_data(block, last_row_mv, last_col_mv)
            residual, bitrate = self.compress_residual(residual, qp=qp, is_sub_block=False)
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
            self.inter_decoder.process(frame_seq, block_seq, 0, residual, row_mv, col_mv, is_sub_block=False, qp=qp)
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


class IntraFrameEncoder(Coder):
    def __init__(self, height, width, config: CodecConfig, current_frame):
        super().__init__(height, width, config)
        self.frame = current_frame
        self.intra_decoder = IntraFrameDecoder(height, width, config)
            
    def get_intra_data(self, block: YuvBlock):
        vertical_ref = self.intra_decoder.frame.get_vertical_ref_block(
            block.row, block.col, block.block_size == self.config.sub_block_size
        )
        vertical_mae = block.get_mae(vertical_ref)
        horizontal_ref = self.intra_decoder.frame.get_horizontal_ref_block(
            block.row, block.col, block.block_size == self.config.sub_block_size
        )
        horizontal_mae = block.get_mae(horizontal_ref)
        if vertical_mae < horizontal_mae:
            return block.get_residual(vertical_ref), 0
        else:
            return block.get_residual(horizontal_ref), 1

    # this function should be idempotent
    def process(self, block: YuvBlock, block_seq: int, use_sub_blocks: bool, qp: int):
        compressed_residual = []
        descriptors = []
        residual_bitrate = 0
        if use_sub_blocks:
            for sub_block_seq, sub_block in enumerate(block.get_sub_blocks()):
                residual, mode = self.get_intra_data(sub_block)
                residual, bitrate = self.compress_residual(residual, qp=qp, is_sub_block=True)
                residual_bitrate += bitrate
                self.intra_decoder.process(block_seq, sub_block_seq, residual, mode, is_sub_block=True, qp=qp)
                compressed_residual.append(residual)
                descriptors.append(mode)
                descriptors.append(1)
        else:
            residual, mode = self.get_intra_data(block)
            residual, bitrate = self.compress_residual(residual, qp=qp, is_sub_block=False)
            residual_bitrate += bitrate
            self.intra_decoder.process(block_seq, 0, residual, mode, is_sub_block=False, qp=qp)
            compressed_residual.append(residual)
            descriptors.append(mode)
            if self.config.VBSEnable:
                descriptors.append(0)

        reconstructed_block = self.intra_decoder.frame.get_block(block.row, block.col, is_sub_block=False)
        distortion = block.get_SAD(reconstructed_block)
        return compressed_residual, descriptors, distortion, residual_bitrate


class VideoEncoder(VideoCoder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)
        self.bitrate = 0
        self.bitrate_controller = BitRateController(height, width, config)
        if self.config.RCflag > 1:
            fist_config = copy.deepcopy(config)
            fist_config.i_Period = -1
            fist_config.RCflag = 0
            fist_config.is_firstpass = True
            self.firstpass_encoder = VideoEncoder(height, width, fist_config)
    def calculate_RDO(self, bitrate, distortion):
        return distortion + self.config.RD_lambda * bitrate

    def process_p_frame(self, frame: ReferenceFrame):
        compressed_residual = []
        descriptors = []
        self.qp_list = []
        last_row_mv, last_col_mv = 0, 0
        frame_encoder = InterFrameEncoder(self.height, self.width, self.previous_frames, self.config)
        frame_bitrate = 0
        self.bitrate_controller.refresh_frame()
        self.per_row_bit = []
        row_bit = 0
        if self.config.is_firstpass:
            self.vbs_token = []
            self.mv_list = []
        for block_seq, block in enumerate(frame.get_blocks()):
            use_sub_blocks = True
            if self.config.RCflag == 3:
                if not self.vbs_token[block_seq]:
                    use_sub_blocks = False
                last_row_mv = self.mv_list[0][0]
                last_col_mv = self.mv_list[0][1]
                self.mv_list.pop(0)
            if block_seq % self.blocks_per_row == 0:
                qp = self.bitrate_controller.get_qp(is_i_frame=False)
                self.bitrate_controller.update_used_rows()
                self.qp_list.append(qp)
            (
                normal_residual,
                normal_descriptors,
                normal_distortion,
                normal_residual_bitrate,
                normal_last_row_mv,
                normal_last_col_mv,
            ) = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=False, qp=qp)
            normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
            if self.config.VBSEnable:
                if use_sub_blocks:
                    (
                        sub_blocks_residual,
                        sub_blocks_descriptors,
                        sub_blocks_distortion,
                        sub_blocks_residual_bitrate,
                        sub_blocks_last_row_mv,
                        sub_blocks_last_col_mv,
                    ) = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=True, qp=qp)
                    sub_blocks_bitrate = sub_blocks_residual_bitrate + len(sub_blocks_descriptors)
                    use_sub_blocks = self.calculate_RDO(normal_bitrate, normal_distortion) > self.calculate_RDO(sub_blocks_bitrate, sub_blocks_distortion)
                    # roll back normal block status
                if not use_sub_blocks:
                    _ = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=False, qp=qp)
            else:
                use_sub_blocks = False
            block_bitrate = 0
            if use_sub_blocks:
                compressed_residual += sub_blocks_residual
                descriptors += sub_blocks_descriptors
                descriptor_bitrate = self.cal_entrophy_bitcount(sub_blocks_descriptors)
                block_bitrate = sub_blocks_residual_bitrate + descriptor_bitrate
                frame_bitrate += block_bitrate
                last_row_mv, last_col_mv = sub_blocks_last_row_mv, sub_blocks_last_col_mv
            else:
                compressed_residual += normal_residual
                descriptors += normal_descriptors
                descriptor_bitrate = self.cal_entrophy_bitcount(normal_descriptors)
                last_row_mv, last_col_mv = normal_last_row_mv, normal_last_col_mv
                block_bitrate = normal_residual_bitrate + descriptor_bitrate
                frame_bitrate += block_bitrate
            if self.config.RCflag > 0:
                self.bitrate_controller.use_bit_count_for_a_frame(block_bitrate)
            if self.config.is_firstpass:
                self.vbs_token.append(use_sub_blocks)
                self.mv_list.append((last_row_mv, last_col_mv))
                row_bit += block_bitrate
                if block_seq >0 and (block_seq+1) % self.blocks_per_row == 0:
                    self.per_row_bit.append(row_bit)
                    row_bit = 0

        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        self.bitrate += frame_bitrate
        self.frame_bitrate = frame_bitrate
        compressed_data = (compressed_residual, compressed_descriptors, self.qp_list, self.frame_seq)
        decoded_frame = frame_encoder.inter_decoder.frame.to_reference_frame()
        self.frame_processed(decoded_frame)
        return compressed_data

    def process_i_frame(self, frame: ReferenceFrame):
        compressed_residual = []
        descriptors = []
        frame_bitrate = 0
        self.qp_list = []
        frame_encoder = IntraFrameEncoder(self.height, self.width, self.config, frame.data)
        self.bitrate_controller.refresh_frame()
        self.per_row_bit = []
        row_bit = 0
        if self.config.is_firstpass:
            self.vbs_token = []
        for block_seq, block in enumerate(frame.get_blocks()):
            if block_seq % self.blocks_per_row == 0:
                qp = self.bitrate_controller.get_qp(is_i_frame=True)
                self.qp_list.append(qp)
                self.bitrate_controller.update_used_rows()
            (
                normal_residual,
                normal_descriptors,
                normal_distortion,
                normal_residual_bitrate,
            ) = frame_encoder.process(block, block_seq, use_sub_blocks=False, qp=qp)
            normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
            use_sub_blocks = True
            if self.config.RCflag == 3:
                if not self.vbs_token[block_seq]:
                    use_sub_blocks = False
            if self.config.VBSEnable:
                if use_sub_blocks:
                    (
                        sub_blocks_residual,
                        sub_blocks_descriptors,
                        sub_blocks_distortion,
                        sub_blocks_residual_bitrate,
                    ) = frame_encoder.process(block, block_seq, use_sub_blocks=True, qp=qp)
                    sub_blocks_bitrate = sub_blocks_residual_bitrate + len(sub_blocks_descriptors)
                    use_sub_blocks = self.calculate_RDO(normal_bitrate, normal_distortion) > self.calculate_RDO(sub_blocks_bitrate, sub_blocks_distortion)
                    # roll back normal block status
                if not use_sub_blocks:
                    _ = frame_encoder.process(block, block_seq, use_sub_blocks=False, qp=qp)
            else:
                use_sub_blocks = False
            block_bitrate = 0
            if use_sub_blocks:
                compressed_residual += sub_blocks_residual
                descriptors += sub_blocks_descriptors
                descriptor_bitrate = self.cal_entrophy_bitcount(sub_blocks_descriptors)
                block_bitrate = sub_blocks_residual_bitrate + descriptor_bitrate
                frame_bitrate += block_bitrate
            else:
                compressed_residual += normal_residual
                descriptors += normal_descriptors
                descriptor_bitrate = self.cal_entrophy_bitcount(normal_descriptors)
                block_bitrate = normal_residual_bitrate + descriptor_bitrate
                frame_bitrate += block_bitrate
            if self.config.RCflag > 0:
                self.bitrate_controller.use_bit_count_for_a_frame(block_bitrate)
            if self.config.is_firstpass:
                self.vbs_token.append(use_sub_blocks)
                row_bit += block_bitrate
                if block_seq >0 and (block_seq+1) % self.blocks_per_row == 0:
                    self.per_row_bit.append(row_bit)
                    row_bit = 0
        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        self.bitrate += frame_bitrate
        self.frame_bitrate = frame_bitrate
        compressed_data = (compressed_residual, compressed_descriptors, self.qp_list, self.frame_seq)
        decoded_frame = frame_encoder.intra_decoder.frame.to_reference_frame()
        self.frame_processed(decoded_frame)
        return compressed_data

    def process(self, frame: ReferenceFrame):
        'if the rc == 2, we need to pre-execute the encoding first and the determine if is i_frame or p_frame'
        if self.config.RCflag > 1:
            if self.frame_seq:
                self.firstpass_encoder.config.qp = sum(self.qp_list) // len(self.qp_list)
            self.firstpass_encoder.frame_seq = self.frame_seq
            self.firstpass_encoder.process(frame)
            first_pass_bit = self.firstpass_encoder.frame_bitrate
            self.bitrate_controller.set_row_ratio(self.firstpass_encoder.per_row_bit)
            if self.firstpass_encoder.frame_seq == 1:
                self.bitrate_controller.update_itable(self.firstpass_encoder.config.qp, first_pass_bit)
            else:
                self.bitrate_controller.update_ptable(self.firstpass_encoder.config.qp, first_pass_bit)
            if self.is_p_frame():
                if first_pass_bit > self.bitrate_controller.threshold_dic[self.firstpass_encoder.config.qp]:
                    self.frame_seq = self.config.i_Period
            if self.config.RCflag == 3:
                self.vbs_token = self.firstpass_encoder.vbs_token
                self.mv_list = self.firstpass_encoder.mv_list
        if self.is_i_frame():
            return self.process_i_frame(frame)
        else:
            return self.process_p_frame(frame)
