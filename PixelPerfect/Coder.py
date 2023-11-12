import numpy as np
from PixelPerfect.Yuv import YuvFrame
from PixelPerfect.ResidualProcessor import ResidualProcessor
from bitstring import BitArray, BitStream
from math import log2, floor


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
        RD_lambda: float = 0,
        VBSEnable: bool = False,
        FMEEnable: bool = False,
        FastME: bool = False,
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
        self.RD_lambda = RD_lambda
        self.VBSEnable = VBSEnable
        self.FMEEnable = FMEEnable
        self.FastME = FastME

class Coder:
    def __init__(self, height, width, config: CodecConfig) -> None:
        conflict = []
        if config.VBSEnable:
            conflict.append("VBSEnable")
        if config.FMEEnable:
            conflict.append("FMEEnable")
        if config.FastME:
            conflict.append("FastME")
        if len(conflict) > 1:
            raise Exception(
                "Error! The following options cannot be enabled at the same time: "
                + ", ".join(conflict)
            )

        if config.do_entropy and (not config.do_dct or not config.do_quantization):
            raise Exception(
                "Error! Entropy coding can only be enabled when DCT and quantization are enabled"
            )
        
        
        self.frame_seq = 0
        self.config = config
        self.height = height
        self.width = width
        self.previous_frame = YuvFrame(
            np.full((self.height, self.width), 128),
            self.config.block_size,
        )
        self.residual_processor = ResidualProcessor(
            self.config.block_size,
            self.config.quant_level,
            self.config.approximated_residual_n,
        )
        self.bitrate = 0



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

    def create_FME_ref(self):
        x, y = self.previous_frame.shape
        self.FME_ref_frame = np.zeros((2 * x - 1, 2 * y - 1))
        for i in range(x):
            for j in range(y):
                self.FME_ref_frame[2 * i, 2 * j] = self.previous_frame.data[i, j]

                # Calculate the average of neighbors and store it in the result array
                if i < x - 1:
                    self.FME_ref_frame[2 * i + 1, 2 * j] = round(
                        self.previous_frame.data[i, j] / 2
                        + self.previous_frame.data[i + 1, j] / 2
                    )
                if j < y - 1:
                    self.FME_ref_frame[2 * i, 2 * j + 1] = round(
                        self.previous_frame.data[i, j] / 2
                        + self.previous_frame.data[i, j + 1] / 2
                    )
                if i < x - 1 and j < y - 1:
                    self.FME_ref_frame[2 * i + 1, 2 * j + 1] = round(
                        self.previous_frame.data[i + 1, j + 1] / 2
                        + self.previous_frame.data[i, j] / 2
                    )

    # region Decoding
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

    def dediagonalize_sequence(self, sequence):
        # put it back to 2d array
        quantized_data = [
            [0 for i in range(self.config.block_size)]
            for j in range(self.config.block_size)
        ]
        i = 0
        for diagonal_length in range(1, self.config.block_size + 1):
            for j in range(diagonal_length):
                quantized_data[j][diagonal_length - j - 1] = sequence[i]
                i += 1
        for diagonal_length in range(self.config.block_size - 1, 0, -1):
            for j in range(diagonal_length):
                quantized_data[self.config.block_size - diagonal_length + j][
                    -1 * j - 1
                ] = sequence[i]
                i += 1
        # retransform the data:
        quantized_data = np.array(quantized_data)
        return quantized_data

    def Entrophy_decoding(self, data):
        data.pos = 0
        RLE_coded = []
        while data.pos != len(data):
            RLE_coded.append(data.read("se"))
        RLE_decoded = self.RLE_decoding(RLE_coded)
        return RLE_decoded

    def decompress_residual(self, residual):
        if self.config.do_entropy:
            residual = self.Entrophy_decoding(residual)
            residual = self.dediagonalize_sequence(residual)
        if self.config.do_quantization:
            residual = self.residual_processor.de_quantization(residual)
        if self.config.do_dct:
            residual = self.residual_processor.de_dct(residual)
        return residual

    def decompress_descriptors(self, descriptors):
        if self.config.do_entropy:
            descriptors = self.Entrophy_decoding(descriptors)
        return descriptors

    # endregion

    # region Encoding
    def cal_entrophy_bitcount(self, sequence):
        sequence = self.RLE_coding(sequence)
        length = 0
        for v in sequence:
            if v == 0:
                length += 1
            else:
                length += 3 + 2 * floor(log2(abs(v)))
        return length

    def RLE_coding(self, data):
        sequence = []
        zero_count = 0
        non_zero_count = 0
        at_last = True
        for v in reversed(data):
            if v == 0:
                if non_zero_count != 0:
                    sequence.append(non_zero_count)
                    non_zero_count = 0
                zero_count += 1
            else:
                if zero_count != 0:
                    if at_last:
                        sequence.append(0)
                        zero_count = 0
                        at_last = False
                    else:
                        sequence.append(zero_count)
                        zero_count = 0
                non_zero_count -= 1
                at_last = False
                sequence.append(v)
        if non_zero_count != 0:
            sequence.append(non_zero_count)
        else:
            sequence.append(zero_count)
        sequence.reverse()
        return sequence

    def diagonalize_matrix(self, data):
        max_col = data.shape[1]
        max_row = data.shape[0]
        fdiag = [[] for _ in range(max_row + max_col - 1)]
        for y in range(max_col):
            for x in range(max_row):
                fdiag[x + y].append(data[y, x])
        sequence = []
        for diag in fdiag:
            sequence.extend(diag)
        return sequence

    def entrophy_coding(self, sequence):
        sequence = self.RLE_coding(sequence)
        bit_sequence = BitStream().join([BitArray(se=i) for i in sequence])
        return bit_sequence

    def compress_residual(self, residual):
        if self.config.do_approximated_residual:
            residual = self.residual_processor.approx(residual)
        if self.config.do_dct:
            residual = self.residual_processor.dct_transform(residual)
        if self.config.do_quantization:
            residual = self.residual_processor.quantization(residual)
        if self.config.do_entropy:
            residual = self.diagonalize_matrix(residual)
            residual = self.entrophy_coding(residual)
            self.bitrate += residual.length
        else:
            self.bitrate += self.cal_entrophy_bitcount(
                self.diagonalize_matrix(residual)
            )
        return residual

    def compress_descriptors(self, descriptors):
        if self.config.do_entropy:
            descriptors = self.entrophy_coding(descriptors)
            self.bitrate += descriptors.length
        else:
            self.bitrate += self.cal_entrophy_bitcount(descriptors)
        return descriptors

    # endregion
