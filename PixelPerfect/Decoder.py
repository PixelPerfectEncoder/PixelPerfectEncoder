from PixelPerfect.Yuv import YuvFrame
from PixelPerfect.Coder import CodecConfig, Coder
import numpy as np

class IntraFrameDecoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame = np.zeros(
            [self.previous_frame.height, self.previous_frame.width], dtype=np.uint8
        )
        self.seq = 0
    
    def process(self, compressed_block_data):
        residual, mode = compressed_block_data
        block_size = self.config.block_size
        row_block_num = self.previous_frame.width // block_size
        row = self.seq // row_block_num * block_size
        col = self.seq % row_block_num * block_size
        residual = self.decompress_residual(residual)
        ref_block = np.full([block_size, block_size], 128)
        if mode == 0: # vertical
            if row != 0:
                ref_row = self.frame[row - 1 : row, col : col + block_size]
                ref_block = np.repeat(ref_row, repeats=block_size, axis=0)
        else: # horizontal
            if col != 0: 
                ref_col = self.frame[row : row + block_size, col - 1 : col]
                ref_block = np.repeat(ref_col, repeats=block_size, axis=1)
        self.frame[row : row + block_size, col : col + block_size] = (
            residual + ref_block
        )
        self.seq += 1

class Decoder(Coder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)

    def process(self, compressed_data):
        if self.is_p_frame():
            frame = np.zeros(
                [self.previous_frame.height, self.previous_frame.width], dtype=np.uint8
            )
            block_size = self.config.block_size
            row_block_num = self.previous_frame.width // block_size
            last_row_mv, last_col_mv = 0, 0
            for seq, data in enumerate(compressed_data):
                row = seq // row_block_num * block_size
                col = seq % row_block_num * block_size
                residual, diff_row_mv, diff_col_mv = data
                row_mv, col_mv = diff_row_mv + last_row_mv, diff_col_mv + last_col_mv
                last_row_mv, last_col_mv = row_mv, col_mv
                residual = self.decompress_residual(residual)
                ref_row = row + row_mv
                ref_col = col + col_mv
                reconstructed_block = (
                    self.previous_frame.data[
                        ref_row : ref_row + block_size, ref_col : ref_col + block_size
                    ]
                    + residual
                )
                frame[row : row + block_size, col : col + block_size] = reconstructed_block
        else:
            intra_decoder = IntraFrameDecoder(self.height, self.width, self.config)
            for data in compressed_data:
                intra_decoder.process(data)
            frame = intra_decoder.frame    
            
        frame = np.clip(frame, 0, 255)
        res = YuvFrame(frame, self.config.block_size)
        self.frame_processed(res)
        return res
