import math
import numpy as np
import bisect
from scipy.fftpack import idct, dct


class ResidualProcessor:
    def init_approx_matrix(self):
        self.max_exp = int(np.log2(self.max_val)) + 1
        self.two_to_n = np.array(
            [2**i for i in range(0, self.max_exp + 1)], dtype=np.int16
        )
        biggest_n = 2**self.max_exp
        self.n_to_two = np.empty(biggest_n + 1, dtype=np.uint16)
        self.quant_matrix = np.zeros((self.block_size, self.block_size))
        for v in range(0, biggest_n + 1):
            pos = bisect.bisect_left(self.two_to_n, v)
            if pos != 0 and v - self.two_to_n[pos - 1] <= self.two_to_n[pos] - v:
                self.n_to_two[v] = pos - 1
            else:
                self.n_to_two[v] = pos

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
        return np.clip(decoded, -self.max_val, self.max_val)

    def approx(self, residual: np.ndarray) -> np.ndarray:
        return self.n_to_two[np.abs(residual)] * np.sign(residual)

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
