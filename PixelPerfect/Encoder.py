from PixelPerfect.Yuv import YuvVideo, YuvFrame, YuvBlock
from PixelPerfect.Common import CodecConfig
from PixelPerfect.Decoder import Decoder
from PixelPerfect.ResidualProcessor import ResidualProcessor
import numpy as np

class Encoder():
    def __init__(self, video : YuvVideo, config : CodecConfig):
        self.video = video
        self.config = config
        self.decoder = Decoder(video.meta, config)
        self.residual_processor = ResidualProcessor()
        
    def is_better_match_block(self, di, dj, block : YuvBlock, min_mae, best_i, best_j) -> bool:
        i = block.row_position + di
        j = block.col_position + dj
        block_size = self.config.block_size
        if (0 <= i <= self.decoder_frame.shape[0] - block_size and
            0 <= j <= self.decoder_frame.shape[1] - block_size):
            reference_block_data = self.decoder_frame.data[i:i + block_size, j:j + block_size]
            mae = block.get_mae(reference_block_data)
            if mae > min_mae:
                return False, None
            if mae < min_mae:
                return True, mae
            if abs(di) + abs(dj) > abs(best_i - block.row_position) + abs(best_j - block.col_position):
                return False, None
            if abs(di) + abs(dj) < abs(best_i - block.row_position) + abs(best_j - block.col_position):
                return True, mae
            if di > best_i - block.row_position:
                return False, None
            if di < best_i - block.row_position:
                return True, mae
            if dj < best_j - block.col_position:
                return True, mae
        return False, None
    
    def find_best_match_block(self, block : YuvBlock) -> YuvBlock:
        min_mae = float('inf')
        best_i, best_j = None, None
        offset = self.config.block_search_offset
        for di in range(-offset, offset + 1):
            for dj in range(-offset, offset + 1):
                is_better_match, mae = self.is_better_match_block(di, dj, block, min_mae, best_i, best_j)
                if is_better_match:
                    min_mae = mae
                    best_i, best_j = block.row_position + di, block.col_position + dj
                            
        block_size = self.config.block_size
        return YuvBlock(
            self.decoder_frame.data[best_i:best_i + block_size, best_j:best_j + block_size],
            block_size,
            best_i,
            best_j
        )
    
    def process(self):
        self.decoder_frame = YuvFrame(np.full((self.video.meta.height, self.video.meta.width), 128))
        self.decoder = Decoder(self.video.meta, self.config)
        for frame in self.video.get_y_frames():
            compressed_data = []
            for block in frame.get_blocks(self.config.block_size):
                best_match_block = self.find_best_match_block(block)
                residual = block.get_residual(best_match_block.data)
                if self.config.do_approximated_residual:
                    residual = self.residual_processor.encode(residual)
                compressed_data.append((
                    best_match_block.row_position, 
                    best_match_block.col_position, 
                    residual))
            
            yield compressed_data
            self.decoder_frame = self.decoder.process(compressed_data)