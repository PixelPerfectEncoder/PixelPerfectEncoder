import math
import numpy as np
from scipy.fft import idctn, dctn
from PixelPerfect.CodecConfig import CodecConfig
from functools import cache

class ResidualProcessor:
    def init_approx_matrix(self):
        divider = 2**self.config.approximated_residual_n
        self.residual2round = np.array(
            [int(i // divider * divider) for i in range(1000)]
        )

    @staticmethod
    def generate_quant_matrix(block_size: int, quant_level: int):
        quant_matrix = np.zeros((block_size, block_size), dtype=np.uint32)
        for iy, ix in np.ndindex(quant_matrix.shape):
            if (ix + iy) < block_size - 1:
                quant_matrix[iy][ix] = math.pow(2, quant_level)
            elif (ix + iy) == block_size - 1:
                quant_matrix[iy][ix] = math.pow(2, quant_level + 1)
            else:
                quant_matrix[iy][ix] = math.pow(2, quant_level + 2)
        return quant_matrix

    @cache
    def get_quant_matrix(self, quant_level: int, is_sub_block: bool):
        if is_sub_block:
            return ResidualProcessor.generate_quant_matrix(self.config.sub_block_size, max(quant_level - 1, 0))
        else:
            return ResidualProcessor.generate_quant_matrix(self.config.block_size, quant_level)
    
    def __init__(self, config: CodecConfig):
        self.config = config
        self.init_approx_matrix()
    
    def approx(self, residual: np.ndarray) -> np.ndarray:
        return self.residual2round[np.abs(residual)] * np.sign(residual)

    def dct_transform(self,residuals):
        transform = dctn(residuals, type=2, norm="ortho")
        # transform = dct(dct(residuals.T, type =2,norm='ortho').T, norm='ortho')
        transform = np.rint(transform)
        return transform

    def quantization(self, dct: np.ndarray, quant_level: int, is_sub_block: bool):
        quant_matrix = self.get_quant_matrix(quant_level, is_sub_block)        
        quantized = np.divide(dct, quant_matrix)
        quantized = np.rint(quantized)
        return quantized

    def de_quantization(self, data: np.ndarray, quant_level: int, is_sub_block: bool):
        quant_matrix = self.get_quant_matrix(quant_level, is_sub_block)
        original = np.multiply(data, quant_matrix)
        return original

    def de_dct(self, data):
        original = idctn(data, type=2, norm="ortho")
        # original = idct(idct(data.T, type =2,norm='ortho').T, norm='ortho')
        # original = np.rint(original)
        return original
