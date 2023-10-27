import math
import numpy as np
import bisect
from scipy.fftpack import idct, dct


class ResidualProcessor:
    def init_approx_matrix(self):
        divider = 2 ** self.approximated_residual_n
        self.residual2round = np.array([int(i // divider * divider) for i in range(1000)])
        print(self.residual2round)
        
    def init_quant_matrix(self):
        self.quant_matrix = np.zeros((self.block_size, self.block_size))
        if self.quant_level < 0:
            raise Exception("Error! the quanti_level must be larger than 0")
        elif self.quant_level > math.log(self.block_size + 7, 2):
            raise Exception("Error! the quanti_level is too large")
        for iy, ix in np.ndindex(self.quant_matrix.shape):
            if (ix + iy) < self.block_size - 1:
                self.quant_matrix[iy][ix] = math.pow(2, self.quant_level)
            elif (ix + iy) == self.block_size - 1:
                self.quant_matrix[iy][ix] = math.pow(2, self.quant_level + 1)
            else:
                self.quant_matrix[iy][ix] = math.pow(2, self.quant_level + 2)

    def __init__(self, block_size, quant_level=2, approximated_residual_n=2):
        self.block_size = block_size
        self.quant_level = quant_level
        self.approximated_residual_n = approximated_residual_n
        self.init_approx_matrix()
        self.init_quant_matrix()

    def approx(self, residual: np.ndarray) -> np.ndarray:
        return self.residual2round[np.abs(residual)] * np.sign(residual)

    def dct_transform(self, residuals):
        transform = dct(dct(residuals.T, norm="ortho").T, norm="ortho")
        return transform

    def quantization(self, dct):
        quantized = np.divide(dct, self.quant_matrix)
        quantized = np.round(quantized)
        return quantized

    def de_quantization(self, data):
        original = np.multiply(data, self.quant_matrix)
        return original

    def de_dct(self, data):
        original = idct(idct(data.T, norm="ortho").T, norm="ortho")
        return original
