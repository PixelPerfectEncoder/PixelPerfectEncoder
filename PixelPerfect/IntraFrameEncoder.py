from PixelPerfect.Yuv import YuvBlock, ConstructingFrame
from PixelPerfect.Coder import Coder
from PixelPerfect.IntraFrameDecoder import IntraFrameDecoder
from PixelPerfect.CodecConfig import CodecConfig

class IntraFrameEncoder(Coder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)
        self.intra_decoder = IntraFrameDecoder(height, width, config)
            
    def get_intra_data(self, block: YuvBlock, constructing_frame):
        if self.config.ParallelMode == 1:
            plain_ref = constructing_frame.get_plain_ref_block(
                block.row, block.col, block.block_size == self.config.sub_block_size
            )
            return block.get_residual(plain_ref), 0
        vertical_ref = constructing_frame.get_vertical_ref_block(
            block.row, block.col, block.block_size == self.config.sub_block_size
        )
        vertical_mae = block.get_mae(vertical_ref)
        horizontal_ref = constructing_frame.get_horizontal_ref_block(
            block.row, block.col, block.block_size == self.config.sub_block_size
        )
        horizontal_mae = block.get_mae(horizontal_ref)
        if vertical_mae < horizontal_mae:
            return block.get_residual(vertical_ref), 0
        else:
            return block.get_residual(horizontal_ref), 1

    # this function should be idempotent
    def process(self, block: YuvBlock, use_sub_blocks: bool, constructing_frame_data):
        constructing_frame = ConstructingFrame(self.config, constructing_frame_data)
        compressed_residual = []
        descriptors = []
        residual_bitrate = 0
        block_seq = self.get_seq_by_position(block.row, block.col)
        if use_sub_blocks:
            for sub_block_seq, sub_block in enumerate(block.get_sub_blocks()):
                residual, mode = self.get_intra_data(sub_block, constructing_frame)
                residual, bitrate = self.compress_residual(residual, self.config.qp, is_sub_block=True)
                residual_bitrate += bitrate
                self.intra_decoder.process(block_seq, sub_block_seq, residual, mode, 
                                           is_sub_block=True, 
                                           constructing_frame_data=constructing_frame_data)
                compressed_residual.append(residual)
                descriptors.append(mode)
                descriptors.append(1)
        else:
            residual, mode = self.get_intra_data(block, constructing_frame)
            residual, bitrate = self.compress_residual(residual, self.config.qp, is_sub_block=False)
            residual_bitrate += bitrate
            self.intra_decoder.process(block_seq, 0, residual, mode, 
                                       is_sub_block=False, 
                                       constructing_frame_data=constructing_frame_data)
            compressed_residual.append(residual)
            descriptors.append(mode)
            if self.config.VBSEnable:
                descriptors.append(0)

        reconstructed_block = self.intra_decoder.frame.get_block(block.row, block.col, is_sub_block=False)
        distortion = block.get_SAD(reconstructed_block)
        return compressed_residual, descriptors, distortion, residual_bitrate


