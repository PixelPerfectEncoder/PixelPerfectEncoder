import numpy as np
from PixelPerfect.Yuv import YuvFrame, YuvInfo
from PixelPerfect.ResidualProcessor import ResidualProcessor


class CodecConfig:
    def __init__(
        self,
        block_size,
        block_search_offset,
        i_Period: int = -1,
        quant_level: int = 2,
        approximated_residual_n: int = 2,
        do_approximated_residual: bool = False,
        do_dct: bool = False,
        do_quantization: bool = False,
        do_entropy: bool = False,
    ) -> None:
        self.block_size = block_size
        self.block_search_offset = block_search_offset
        self.i_Period = i_Period
        self.quant_level = quant_level
        self.approximated_residual_n = approximated_residual_n
        self.do_approximated_residual = do_approximated_residual
        self.do_dct = do_dct
        self.do_quantization = do_quantization
        self.do_entropy = do_entropy


class Coder:
    def __init__(self, video_info: YuvInfo, config: CodecConfig) -> None:
        self.frame_seq = 0
        self.config = config
        self.video_info = video_info
        self.previous_frame = YuvFrame(
            np.full((self.video_info.height, self.video_info.width), 128),
            self.config.block_size,
        )
        self.residual_processor = ResidualProcessor(
            self.config.block_size, self.config.quant_level, self.config.approximated_residual_n
        )

    def is_p_frame(self):
        if self.config.i_Period == -1:
            return True
        if self.config.i_Period == 0:
            return False
        if self.frame_seq % self.config.i_Period == 0:
            return False
        else:
            return True

    def frame_processed(self, frame):
        self.frame_seq += 1
        self.previous_frame = frame

    def is_i_frame(self):
        return not self.is_p_frame()
    
    def decompress_residual(self, residual):
        if self.config.do_entropy:
            residual = self.Entrophy_decoding(residual)
        if self.config.do_quantization:
            residual = self.residual_processor.de_quantization(residual)
        if self.config.do_dct:
            residual = self.residual_processor.de_dct(residual)
        return residual
    
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
            RLE_coded.append(data.read("se"))
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