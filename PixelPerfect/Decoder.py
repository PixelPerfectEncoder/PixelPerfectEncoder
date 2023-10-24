from PixelPerfect.Yuv import YuvMeta, YuvFrame
from PixelPerfect.Common import CodecConfig
from PixelPerfect.ResidualProcessor import ResidualProcessor
import numpy as np

class Decoder:
    def __init__(self, yuv_info : YuvMeta, config : CodecConfig):
        self.yuv_info = yuv_info
        self.config = config
        self.previous_frame = YuvFrame(np.full((self.yuv_info.height, self.yuv_info.width), 128))
        self.residual_processor = ResidualProcessor(config.block_size)
    def RLE_decoding(self, sequence, data):
        decoded = []
        index = 0
        try:
            while len(decoded) < pow(self.config.block_size,2) :
                if sequence[index] == 0:
                    #while len(decoded) < pow(self.config.block_size,2):
                    while len(decoded) < pow(self.config.block_size,2):
                        decoded.append(0)
                elif sequence[index] < 0:
                    for i in range(abs(sequence[index])):
                        index+=1
                        decoded.append(sequence[index])
                else:
                    for i in range(abs(sequence[index])):
                        decoded.append(0)
                index += 1
        except:
            print(data)
            print(sequence)
            print(index)
            print(decoded)
        return decoded

    def Entrophy_decoding(self, data):
        RLE_coded = []
        for bit in data:
            RLE_coded.append(bit.se)
        RLE_decoded = self.RLE_decoding(RLE_coded, data)
        #put it back to 2d array
        quantized_data = [[0 for i in range(self.config.block_size)] for j in range(self.config.block_size)]
        i = 0
        for diagonal_length in range(1,self.config.block_size+1):
            for j in range(diagonal_length):
                 quantized_data[j][diagonal_length - j - 1] = RLE_decoded[i];
                 i+=1
        for diagonal_length in range(self.config.block_size-1,0, -1):
            for j in range(diagonal_length):
                 quantized_data[self.config.block_size - diagonal_length + j][-1*j] = RLE_decoded[i];
                 i+=1
        #retransform the data:
        dct_data = self.residual_processor.de_quantization(quantized_data)
        residual = self.residual_processor.de_dct(dct_data)
        return residual
    def process(self, data):
        frame = np.zeros([self.yuv_info.height, self.yuv_info.width], dtype=np.uint8)
        block_size = self.config.block_size
        row_block_num = self.yuv_info.width // block_size
        for seq, block_data in enumerate(data):
            ref_row, ref_col, residual, Entrophy_coded_data = block_data
            residual = self.Entrophy_decoding(Entrophy_coded_data)
            row = seq // row_block_num * block_size
            col = seq % row_block_num * block_size
            if self.config.do_approximated_residual:
                residual = self.residual_processor.decode(residual)
            frame[row:row + block_size, col:col + block_size] \
                = self.previous_frame.data[ref_row:ref_row + block_size, ref_col:ref_col + block_size] + residual
        self.previous_frame = YuvFrame(frame)
        return self.previous_frame

        