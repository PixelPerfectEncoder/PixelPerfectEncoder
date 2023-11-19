import numpy as np
from PixelPerfect.Yuv import YuvFrame, ReferenceFrame
from PixelPerfect.ResidualProcessor import ResidualProcessor
from PixelPerfect.CodecConfig import CodecConfig
from bitstring import BitArray, BitStream
from math import log2, floor
from typing import Deque

class Coder:
    def __init__(self, height, width, config: CodecConfig) -> None:
        if config.do_entropy and (not config.do_dct or not config.do_quantization):
            raise Exception(
                "Error! Entropy coding can only be enabled when DCT and quantization are enabled"
            )
        if config.do_quantization and not config.do_dct:
            raise Exception(
                "Error! Quantization can only be enabled when DCT is enabled"
            )
            
        self.config = config
        self.height = height
        self.width = width
        _, padded_width = YuvFrame.get_padded_size(height, width, config.block_size)
        self.row_block_num = padded_width // self.config.block_size
        self.residual_processor = ResidualProcessor(
            self.config.block_size,
            self.config.quant_level,
            self.config.approximated_residual_n,
        )
        
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

    def decompress_residual(self, residual, block_size):
        if self.config.do_entropy:
            residual = self.Entrophy_decoding(residual, block_size)
            residual = self.dediagonalize_sequence(residual, block_size)
        if self.config.do_quantization:
            residual = self.residual_processor.de_quantization(residual)
        if self.config.do_dct:
            residual = self.residual_processor.de_dct(residual)
        return residual

    def decompress_descriptors(self, descriptors):
        if self.config.do_entropy:
            descriptors = self.Entrophy_decoding(descriptors)
        return descriptors

    def get_position_by_seq(self, block_seq, sub_block_seq):
        row = block_seq // self.row_block_num * self.config.block_size
        col = block_seq % self.row_block_num * self.config.block_size
        row += (sub_block_seq // 2) * self.config.sub_block_size
        col += (sub_block_seq % 2) * self.config.sub_block_size
        return row, col

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
        bitrate = 0
        if self.config.do_approximated_residual:
            residual = self.residual_processor.approx(residual)
        if self.config.do_dct:
            residual = self.residual_processor.dct_transform(residual)
        if self.config.do_quantization:
            residual = self.residual_processor.quantization(residual)
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


class VideoCoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame_seq = 0
        self.previous_frames: Deque[ReferenceFrame] = Deque(maxlen=config.nRefFrames)
        self.previous_frames.append(ReferenceFrame(config, np.full(shape=(self.height, self.width), fill_value=128, dtype=np.uint8)))
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
        if self.is_i_frame():
            self.previous_frames.clear()
        self.previous_frames.append(frame)
        
    def is_i_frame(self):
        return not self.is_p_frame()
    
    