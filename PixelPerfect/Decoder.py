from PixelPerfect.Yuv import YuvMeta, YuvFrame
from PixelPerfect.Common import EncoderConfig
from PixelPerfect.ResidualProcessor import ResidualProcessor
import numpy as np

class Decoder:
    def __init__(self, yuv_info : YuvMeta, encoder_config : EncoderConfig):
        self.yuv_info = yuv_info
        self.encoder_config = encoder_config
        self.previous_frame = YuvFrame(np.full((self.yuv_info.height, self.yuv_info.width), 128))
        self.residual_processor = ResidualProcessor()
        
    def process(self, data):
        frame = np.zeros([self.yuv_info.height, self.yuv_info.width], dtype=np.uint8)
        block_size = self.encoder_config.block_size
        row_block_num = self.yuv_info.width // block_size
        for seq, block_data in enumerate(data):
            ref_row, ref_col, residual = block_data
            row = seq // row_block_num * block_size
            col = seq % row_block_num * block_size
            
            frame[row:row + block_size, col:col + block_size] \
                = self.previous_frame.data[ref_row:ref_row + block_size, ref_col:ref_col + block_size] \
                    + self.residual_processor.decode(residual)
        self.previous_frame = YuvFrame(frame)
        return self.previous_frame

        