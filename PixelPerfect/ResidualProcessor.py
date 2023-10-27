import math
import numpy as np
import bisect
from scipy.fftpack import idct, dct


class ResidualProcessor:
    def init_approx_matrix(self):
        self.two_to_n = [2**i for i in range(1, 4)]
        print(self.two_to_n)
        self.n_to_two = [2 for _ in range(500)]
        for v1 in range(600):
            for i, v2 in enumerate(self.two_to_n):
                if v1 <= v2:
                    if i == 0 or v1 - self.two_to_n[i - 1] > v2 - v1:
                        self.n_to_two[v1] = i
                    else:
                        self.n_to_two[v1] = i - 1
                    break
        self.two_to_n = np.array(self.two_to_n)
        self.n_to_two = np.array(self.n_to_two)

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

    def __init__(self, block_size, max_val=255, quant_level=2):
        self.block_size = block_size
        self.max_val = max_val
        self.quant_level = quant_level
        self.init_approx_matrix()
        self.init_quant_matrix()

    def de_approx(self, residual: np.ndarray) -> np.ndarray:
        decoded = self.two_to_n[np.abs(residual)] * np.sign(residual)
        return decoded

    def approx(self, residual: np.ndarray) -> np.ndarray:
        encoded = self.n_to_two[np.abs(residual)] * np.sign(residual)
        return encoded

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
