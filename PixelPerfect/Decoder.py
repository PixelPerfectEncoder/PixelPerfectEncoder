from PixelPerfect.Yuv import YuvInfo, YuvFrame
from PixelPerfect.Coder import CodecConfig, Coder
import numpy as np


class Decoder(Coder):
    def __init__(self, video_info: YuvInfo, config: CodecConfig):
        super().__init__(video_info, config)

    def RLE_decoding(self, sequence):
        decoded = []
        index = 0
        while index < len(sequence):
            if sequence[index] < 0:
                decoded.extend(sequence[index + 1 : index + 1 - sequence[index]])
                index -= sequence[index]
            else:
                decoded.extend([0] * sequence[index])
            index += 1
        decoded.extend([0] * (pow(self.config.block_size, 2) - len(decoded)))
        return decoded

    def Entrophy_decoding(self, data):
        data.pos = 0
        RLE_coded = []
        while data.pos != len(data):
            RLE_coded.append(data.read('se'))
        RLE_decoded = self.RLE_decoding(RLE_coded)
        # put it back to 2d array
        quantized_data = [
            [0 for i in range(self.config.block_size)]
            for j in range(self.config.block_size)
        ]
        i = 0
        for diagonal_length in range(1, self.config.block_size + 1):
            for j in range(diagonal_length):
                quantized_data[j][diagonal_length - j - 1] = RLE_decoded[i]
                i += 1
        for diagonal_length in range(self.config.block_size - 1, 0, -1):
            for j in range(diagonal_length):
                quantized_data[self.config.block_size - diagonal_length + j][
                    -1 * j - 1
                ] = RLE_decoded[i]
                i += 1
        # retransform the data:
        quantized_data = np.array(quantized_data)
        return quantized_data

    def decouple_block_data(self, row, col, block_data):
        if self.is_p_frame():
            row_mv, col_mv, residual = block_data
            ref_row = row - row_mv
            ref_col = col - col_mv
        else:
            mode, residual = block_data
            if mode == 1:
                ref_col = col
                if row == 0:
                    ref_row = row
                else:
                    ref_row = row - self.config.block_size
            else:
                ref_row = row
                if col == 0:
                    ref_col = col
                else:
                    ref_col = col - self.config.block_size
        return ref_row, ref_col, residual
    
    def process(self, data):
        frame = np.zeros(
            [self.previous_frame.height, self.previous_frame.width], dtype=np.uint8
        )
        block_size = self.config.block_size
        row_block_num = self.previous_frame.width // block_size
        for seq, block_data in enumerate(data):
            row = seq // row_block_num * block_size
            col = seq % row_block_num * block_size
            ref_row, ref_col, residual = self.decouple_block_data(row, col, block_data)    
            if self.config.do_entropy:
                residual = self.Entrophy_decoding(residual)
            if self.config.do_quantization:
                residual = self.residual_processor.de_quantization(residual)
            if self.config.do_dct:
                residual = self.residual_processor.de_dct(residual)
            if self.is_p_frame():
                ref_frame_data = self.previous_frame.data
            else:
                ref_frame_data = frame
            frame[row : row + block_size, col : col + block_size] = (
                ref_frame_data[
                    ref_row : ref_row + block_size, ref_col : ref_col + block_size
                ]
                + residual
            )
        frame = np.clip(frame, 0, 255)
        res = YuvFrame(frame, self.config.block_size)
        self.frame_processed(res)
        return res
