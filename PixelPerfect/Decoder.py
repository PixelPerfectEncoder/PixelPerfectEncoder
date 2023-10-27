from PixelPerfect.Yuv import YuvMeta, YuvFrame
from PixelPerfect.Common import CodecConfig
from PixelPerfect.ResidualProcessor import ResidualProcessor
import numpy as np


class Decoder:
    def __init__(self, yuv_info: YuvMeta, config: CodecConfig):
        self.yuv_info = yuv_info
        self.config = config
        self.previous_frame = YuvFrame(
            np.full((self.yuv_info.height, self.yuv_info.width), 128)
        )
        self.residual_processor = ResidualProcessor(config.block_size)
    
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
        length = data.read("se")
        RLE_coded = data.readlist(str(length) + "*se")
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

    def process(self, data):
        frame = np.zeros([self.yuv_info.height, self.yuv_info.width], dtype=np.uint8)
        block_size = self.config.block_size
        row_block_num = self.yuv_info.width // block_size
        for seq, block_data in enumerate(data):
            ref_row, ref_col, residual = block_data
            if self.config.do_entropy:
                residual = self.Entrophy_decoding(residual)
            if self.config.do_quantization:
                residual = self.residual_processor.de_quantization(residual)
            if self.config.do_dct:
                residual = self.residual_processor.de_dct(residual)
            if self.config.do_approximated_residual:
                residual = self.residual_processor.de_approx(residual)
            row = seq // row_block_num * block_size
            col = seq % row_block_num * block_size
            frame[row : row + block_size, col : col + block_size] = (
                self.previous_frame.data[
                    ref_row : ref_row + block_size, ref_col : ref_col + block_size
                ]
                + residual
            )
        self.previous_frame = YuvFrame(frame)
        return self.previous_frame
