from .Yuv import YuvVideo, YuvFrame

class EncoderConfig:
    def __init__(self, block_size, block_search_offset) -> None:
        self.block_size = block_size
        self.block_search_offset = block_search_offset
    
class Encoder():
    def __init__(video : YuvVideo, encoder_config : EncoderConfig):
        self.video = video
        self.encoder_config = encoder_config
        self.decoder = Decoder(video.meta, encoder_config)
        
    def find_best_match(self, block : YuvBlock) -> YuvBlock:
        min_mae = float('inf')
        best_i, best_j = None, None
        for di in range(-offset, offset + 1):
            for dj in range(-offset, offset + 1):
                i = block.row_position + di
                j = block.col_position + dj
                if (0 <= i <= self.decoder_frame.shape[0] - self.hight and
                    0 <= j <= self.decoder_frame.shape[1] - self.width):
                    reference_block_data = self.decoder_frame.data[i:i + self.hight, j:j + self.width]
                    mae = block.get_mae(reference_block_data)
                    if mae < min_mae:
                        min_mae = mae
                        best_i, best_j = i, j
        
        return YuvBlock(
            self.decoder_frame.data[best_i:best_i + self.hight, best_j:best_j + self.width],
            self.block_size,
            best_i,
            best_j
        )
        
    def process():
        self.decoder_frame = YuvFrame(np.full((self.video.meta.hight, self.video.meta.width), 128))
        for frame in self.video.get_y_frames():
            compressed_data = []
            for block in frame.get_blocks(self.encoder_config.block_size):
                best_match_block = self.find_best_match(block)
                residual = block.get_residual(best_match_block.data)
                compressed_data.append((
                    best_match_block.row_position, 
                    best_match_block.col_position, 
                    residual))
            yield compressed_data
            self.decoder_frame = decoder.process(compressed_bytes)