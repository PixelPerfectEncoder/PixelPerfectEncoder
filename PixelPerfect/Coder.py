import numpy as np
from PixelPerfect.Yuv import YuvFrame
from PixelPerfect.ResidualProcessor import ResidualProcessor
from PixelPerfect.CodecConfig import CodecConfig
from bitstring import BitArray, BitStream
from math import log2, floor

class Coder:
    def __init__(self, height, width, config: CodecConfig) -> None:
        if config.FastME and config.FastME_LIMIT == -1:
            raise Exception(
                "Error! FastME_LIMIT must be set when FastME is enabled"
            )
        
        if config.ParallelMode == 3 and config.nRefFrames != 1:
            raise Exception(
                "Error! nRefFrames must be 1 when ParallelMode is 3"
            )
        
        if config.ParallelMode == 3 and config.num_processes != 2:
            raise Exception(
                "Error! num_processes must be 2 when ParallelMode is 3"
            )
        
        if config.i_Period <= 0:
            raise Exception(
                "Error! i_Period must be positive"
            )
        
        config.need_display = config.DisplayBlocks or config.DisplayMvAndMode or config.DisplayRefFrames
            
        self.config = config
        self.height = height
        self.width = width
        self.row_block_num = width // self.config.block_size
        self.residual_processor = ResidualProcessor(self.config)
        
    # region Decoding
    def RLE_decoding(self, sequence, block_size):
        decoded = []
        index = 0
        while index < len(sequence):
            if sequence[index] < 0:
                decoded.extend(sequence[index + 1 : index + 1 - sequence[index]])
                index -= sequence[index]
            else:
                decoded.extend([0] * sequence[index])
            index += 1
        decoded.extend([0] * (pow(block_size, 2) - len(decoded)))
        return decoded

    def dediagonalize_sequence(self, sequence, block_size):
        # put it back to 2d array
        quantized_data = [
            [0 for i in range(block_size)]
            for j in range(block_size)
        ]
        i = 0
        for diagonal_length in range(1, block_size + 1):
            for j in range(diagonal_length):
                quantized_data[j][diagonal_length - j - 1] = sequence[i]
                i += 1
        for diagonal_length in range(block_size - 1, 0, -1):
            for j in range(diagonal_length):
                quantized_data[block_size - diagonal_length + j][
                    -1 * j - 1
                ] = sequence[i]
                i += 1
        # retransform the data:
        quantized_data = np.array(quantized_data)
        return quantized_data

    def Entrophy_decoding(self, data, block_size):
        data.pos = 0
        RLE_coded = []
        while data.pos != len(data):
            RLE_coded.append(data.read("se"))
        RLE_decoded = self.RLE_decoding(RLE_coded, block_size)
        return RLE_decoded

    def decompress_residual(self, residual: np.ndarray, qp: int, is_sub_block: bool):
        if self.config.do_entropy:
            block_size = self.config.sub_block_size if is_sub_block else self.config.block_size
            residual = self.Entrophy_decoding(residual, block_size)
            residual = self.dediagonalize_sequence(residual, block_size)
        residual = self.residual_processor.de_quantization(residual, qp, is_sub_block)
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

    def compress_residual(self, residual: np.ndarray, qp: int, is_sub_block: int):
        bitrate = 0
        if self.config.do_approximated_residual:
            residual = self.residual_processor.approx(residual)
        residual = self.residual_processor.dct_transform(residual)
        residual = self.residual_processor.quantization(residual, qp, is_sub_block)
        if self.config.do_entropy:
            residual = self.diagonalize_matrix(residual)
            residual = self.entrophy_coding(residual)
            bitrate += residual.length
        else:
            bitrate += self.cal_entrophy_bitcount(
                self.diagonalize_matrix(residual)
            )
        return residual, bitrate

    def compress_descriptors(self, descriptors):
        bitrate = 0
        if self.config.do_entropy:
            descriptors = self.entrophy_coding(descriptors)
            bitrate += descriptors.length
        else:
            bitrate += self.cal_entrophy_bitcount(descriptors)
        return descriptors, bitrate

    # endregion
    