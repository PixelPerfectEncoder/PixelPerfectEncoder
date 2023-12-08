from PixelPerfect.Yuv import ReferenceFrame
from PixelPerfect.VideoCoder import VideoCoder
from PixelPerfect.VideoDecoder import VideoDecoder
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.BitRateController import BitRateController
from PixelPerfect.InterFrameEncoder import InterFrameEncoder
from PixelPerfect.IntraFrameEncoder import IntraFrameEncoder
from PixelPerfect.CompressedFrameData import CompressedFrameData
import multiprocessing

pool = None

class VideoEncoder(VideoCoder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)
        self.bitrate = 0
        self.bitrate_controller = BitRateController(height, width, config)
        self.video_decoder = VideoDecoder(height, width, config)
        global pool
        pool = multiprocessing.Pool(processes=self.config.num_processes)
    def calculate_RDO(self, bitrate, distortion):
        return distortion + self.config.RD_lambda * bitrate

    def p_frame_compare_sub_block_and_normal_block_get_result(self, args):
        frame_encoder, block, last_row_mv, last_col_mv, qp = args
        (
            normal_residual,
            normal_descriptors,
            normal_distortion,
            normal_residual_bitrate,
            normal_last_row_mv,
            normal_last_col_mv,
        ) = frame_encoder.process(block, last_row_mv, last_col_mv, use_sub_blocks=False, qp=qp)
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
            ) = frame_encoder.process(block, last_row_mv, last_col_mv, use_sub_blocks=True, qp=qp)
            sub_blocks_bitrate = sub_blocks_residual_bitrate + len(sub_blocks_descriptors)
            use_sub_blocks = self.calculate_RDO(normal_bitrate, normal_distortion) > self.calculate_RDO(sub_blocks_bitrate, sub_blocks_distortion)
            # roll back normal block status
            if not use_sub_blocks:
                _ = frame_encoder.process(block, last_row_mv, last_col_mv, use_sub_blocks=False, qp=qp)

        if use_sub_blocks:
            return (
                sub_blocks_residual,
                sub_blocks_descriptors,
                sub_blocks_residual_bitrate,
                sub_blocks_last_row_mv,
                sub_blocks_last_col_mv,
            )
        else:
            return (
                normal_residual,
                normal_descriptors,
                normal_residual_bitrate,
                normal_last_row_mv,
                normal_last_col_mv,
            )

    def process_p_frame(self, frame: ReferenceFrame) -> CompressedFrameData:
        compressed_residual = []
        descriptors = []
        last_row_mv, last_col_mv = 0, 0
        frame_encoder = InterFrameEncoder(self.height, self.width, self.previous_frames, self.config)
        qp = self.bitrate_controller.get_qp(is_i_frame=False)
        frame_bitrate = 0
        if self.config.ParallelMode == 0:
            for block in frame.get_blocks():
                (
                    part_compressed_residual, 
                    part_descriptors, 
                    bitrate, 
                    last_row_mv, 
                    last_col_mv
                ) = self.p_frame_compare_sub_block_and_normal_block_get_result(args=(frame_encoder, block, last_row_mv, last_col_mv, qp))
                compressed_residual += part_compressed_residual
                descriptors += part_descriptors
                frame_bitrate += bitrate
        elif self.config.ParallelMode == 1:
            task_parameters = []
            for block in frame.get_blocks():
                task_parameters.append((frame_encoder, block, 0, 0, qp))
            results = pool.map(self.p_frame_compare_sub_block_and_normal_block_get_result, task_parameters)
            for result in results:
                compressed_residual += result[0]
                descriptors += result[1]
                frame_bitrate += result[2]
        elif self.config.ParallelMode == 2:
            pass
        else:
            raise Exception("Unknown ParallelMode")

        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        frame_bitrate += descriptors_bitrate
        if self.config.RCflag == 1:
            self.bitrate_controller.use_bit_count_for_a_frame(frame_bitrate)
        self.bitrate += frame_bitrate
        compressed_data = CompressedFrameData(compressed_residual, compressed_descriptors, qp)        
        decoded_frame = self.video_decoder.process(compressed_data)
        self.frame_processed(decoded_frame)
        return compressed_data

    def iframe_compare_sub_block_and_normal_block_get_result(self, args):
        frame_encoder, block, qp = args
        (
            normal_residual,
            normal_descriptors,
            normal_distortion,
            normal_residual_bitrate,
        ) = frame_encoder.process(block, use_sub_blocks=False, qp=qp)
        normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
        use_sub_blocks = False
        if self.config.VBSEnable:
            (
                sub_blocks_residual,
                sub_blocks_descriptors,
                sub_blocks_distortion,
                sub_blocks_residual_bitrate,
            ) = frame_encoder.process(block, use_sub_blocks=True, qp=qp)
            sub_blocks_bitrate = sub_blocks_residual_bitrate + len(sub_blocks_descriptors)
            use_sub_blocks = self.calculate_RDO(normal_bitrate, normal_distortion) > self.calculate_RDO(sub_blocks_bitrate, sub_blocks_distortion)
            # roll back normal block status
            if not use_sub_blocks:
                _ = frame_encoder.process(block, use_sub_blocks=False, qp=qp)
        if use_sub_blocks:
            return (
                sub_blocks_residual,
                sub_blocks_descriptors,
                sub_blocks_residual_bitrate,
            )
        else:
            return (
                normal_residual,
                normal_descriptors,
                normal_residual_bitrate,
            )

    def process_i_frame(self, frame: ReferenceFrame):
        compressed_residual = []
        descriptors = []
        qp = self.bitrate_controller.get_qp(is_i_frame=True)
        frame_bitrate = 0
        frame_encoder = IntraFrameEncoder(self.height, self.width, self.config, frame.data)
        if self.config.ParallelMode == 0:
            for block in frame.get_blocks():
                (
                    part_residual, 
                    part_descriptors, 
                    bitrate
                ) = self.iframe_compare_sub_block_and_normal_block_get_result(args=(frame_encoder, block, qp))
                compressed_residual += part_residual
                descriptors += part_descriptors
                frame_bitrate += bitrate
        elif self.config.ParallelMode == 1:
            task_parameters = []
            for block in frame.get_blocks():
                task_parameters.append((frame_encoder, block, qp))
            results = pool.map(self.iframe_compare_sub_block_and_normal_block_get_result, task_parameters)
            for result in results:
                compressed_residual += result[0]
                descriptors += result[1]
                frame_bitrate += result[2]
        elif self.config.ParallelMode == 2:
            position_results_pairs = []
            for blocks in frame.get_batch_of_blocks_by_diagonal():
                task_parameters = [(frame_encoder, block, qp) for block in blocks]
                results = pool.map(self.iframe_compare_sub_block_and_normal_block_get_result, task_parameters)
                for block, result in zip(blocks, results):
                    position_results_pairs.append(((block.row, block.col), result))
            position_results_pairs.sort(key=lambda x: x[0])
            for _, result in position_results_pairs:
                compressed_residual += result[0]
                descriptors += result[1]
                frame_bitrate += result[2]
        else:
            raise Exception("Unknown ParallelMode")
        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        frame_bitrate += descriptors_bitrate
        if self.config.RCflag == 1:
            self.bitrate_controller.use_bit_count_for_a_frame(frame_bitrate)
        self.bitrate += frame_bitrate
        compressed_data = CompressedFrameData(compressed_residual, compressed_descriptors, qp)
        decoded_frame = self.video_decoder.process(compressed_data)
        self.frame_processed(decoded_frame)
        return compressed_data

    def process(self, frame: ReferenceFrame):
        if self.is_i_frame():
            return self.process_i_frame(frame)
        else:
            return self.process_p_frame(frame)