from PixelPerfect.Yuv import ReferenceFrame
from PixelPerfect.VideoCoder import VideoCoder
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.BitRateController import BitRateController
from PixelPerfect.InterFrameEncoder import InterFrameEncoder
from PixelPerfect.IntraFrameEncoder import IntraFrameEncoder

class VideoEncoder(VideoCoder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)
        self.bitrate = 0
        self.bitrate_controller = BitRateController(height, width, config)

    def calculate_RDO(self, bitrate, distortion):
        return distortion + self.config.RD_lambda * bitrate

    def process_p_frame(self, frame: ReferenceFrame):
        compressed_residual = []
        descriptors = []
        last_row_mv, last_col_mv = 0, 0
        frame_encoder = InterFrameEncoder(self.height, self.width, self.previous_frames, self.config)
        qp = self.bitrate_controller.get_qp(is_i_frame=False)
        frame_bitrate = 0
        for block_seq, block in enumerate(frame.get_blocks()):
            (
                normal_residual,
                normal_descriptors,
                normal_distortion,
                normal_residual_bitrate,
                normal_last_row_mv,
                normal_last_col_mv,
            ) = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=False, qp=qp)
            normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
            use_sub_blocks = False
            if self.config.VBSEnable:
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

            if use_sub_blocks:
                compressed_residual += sub_blocks_residual
                descriptors += sub_blocks_descriptors
                frame_bitrate += sub_blocks_residual_bitrate
                last_row_mv, last_col_mv = sub_blocks_last_row_mv, sub_blocks_last_col_mv
            else:
                compressed_residual += normal_residual
                descriptors += normal_descriptors
                frame_bitrate += normal_residual_bitrate
                last_row_mv, last_col_mv = normal_last_row_mv, normal_last_col_mv

        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        frame_bitrate += descriptors_bitrate
        if self.config.RCflag == 1:
            self.bitrate_controller.use_bit_count_for_a_frame(frame_bitrate)
        self.bitrate += frame_bitrate
        compressed_data = (compressed_residual, compressed_descriptors, qp)
        decoded_frame = frame_encoder.inter_decoder.frame.to_reference_frame()
        self.frame_processed(decoded_frame)
        return compressed_data

    def process_i_frame(self, frame: ReferenceFrame):
        compressed_residual = []
        descriptors = []
        qp = self.bitrate_controller.get_qp(is_i_frame=True)
        frame_bitrate = 0
        frame_encoder = IntraFrameEncoder(self.height, self.width, self.config, frame.data)
        for block_seq, block in enumerate(frame.get_blocks()):
            (
                normal_residual,
                normal_descriptors,
                normal_distortion,
                normal_residual_bitrate,
            ) = frame_encoder.process(block, block_seq, use_sub_blocks=False, qp=qp)
            normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
            use_sub_blocks = False
            if self.config.VBSEnable:
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
            if use_sub_blocks:
                compressed_residual += sub_blocks_residual
                descriptors += sub_blocks_descriptors
                frame_bitrate += sub_blocks_residual_bitrate
            else:
                compressed_residual += normal_residual
                descriptors += normal_descriptors
                frame_bitrate += normal_residual_bitrate
        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        frame_bitrate += descriptors_bitrate
        if self.config.RCflag == 1:
            self.bitrate_controller.use_bit_count_for_a_frame(frame_bitrate)
        self.bitrate += frame_bitrate
        compressed_data = (compressed_residual, compressed_descriptors, qp)
        decoded_frame = frame_encoder.intra_decoder.frame.to_reference_frame()
        self.frame_processed(decoded_frame)
        return compressed_data

    def process(self, frame: ReferenceFrame):
        if self.is_i_frame():
            return self.process_i_frame(frame)
        else:
            return self.process_p_frame(frame)