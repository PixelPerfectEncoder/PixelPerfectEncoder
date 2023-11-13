import math
import numpy as np
from scipy.fft import idctn, dctn

class ResidualProcessor:
    def init_approx_matrix(self):
        divider = 2**self.approximated_residual_n
        self.residual2round = np.array(
            [int(i // divider * divider) for i in range(1000)]
        )

    def get_quant_matrix(self, block_size, quant_level):
        quant_matrix = np.zeros((block_size, block_size))
        for iy, ix in np.ndindex(quant_matrix.shape):
            if (ix + iy) < block_size - 1:
                quant_matrix[iy][ix] = math.pow(2, quant_level)
            elif (ix + iy) == block_size - 1:
                quant_matrix[iy][ix] = math.pow(2, quant_level + 1)
            else:
                quant_matrix[iy][ix] = math.pow(2, quant_level + 2)
        return quant_matrix

    def __init__(self, block_size, quant_level, approximated_residual_n):
        self.approximated_residual_n = approximated_residual_n
        self.init_approx_matrix()
        self.quant_matrix = self.get_quant_matrix(block_size, quant_level)
        self.sub_block_quant_matrix = self.get_quant_matrix(block_size // 2, max(quant_level - 1, 0))
        
    def approx(self, residual: np.ndarray) -> np.ndarray:
        return self.residual2round[np.abs(residual)] * np.sign(residual)

    def dct_transform(self, residuals):
        transform = dctn(residuals, norm="ortho")
        return transform

    def quantization(self, dct: np.ndarray):
        if dct.shape[0] == self.quant_matrix.shape[0]:
            quant_matrix = self.quant_matrix
        elif dct.shape[0] == self.sub_block_quant_matrix.shape[0]:
            quant_matrix = self.sub_block_quant_matrix
        else:
            raise Exception("Error! Invalid data shape")
        quantized = np.divide(dct, quant_matrix)
        quantized = np.round(quantized)
        return quantized

    def de_quantization(self, data: np.ndarray):
        if data.shape[0] == self.quant_matrix.shape[0]:
            quant_matrix = self.quant_matrix
        elif data.shape[0] == self.sub_block_quant_matrix.shape[0]:
            quant_matrix = self.sub_block_quant_matrix
        else:
            raise Exception("Error! Invalid data shape")
        original = np.multiply(data, quant_matrix)
        return original

    def de_dct(self, data):
        original = idctn(data, norm="ortho")
        return original
