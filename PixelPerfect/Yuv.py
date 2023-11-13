import numpy as np
import cv2
from math import log10, sqrt


class YuvBlock:
    def __init__(self, data, block_size, row, col) -> None:
        self.data = data
        self.block_size = block_size
        self.row_position = row
        self.col_position = col

    def get_mae(self, reference_data: np.ndarray) -> float:
        return np.mean(np.abs(self.data.astype(np.int16) - reference_data.astype(np.int16)))

    def get_SAD(self, reference_data: np.ndarray) -> float:
        return np.sum(np.abs(self.data.astype(np.int16) - reference_data.astype(np.int16)))

    def get_residual(self, reference_data: np.ndarray) -> np.ndarray:
        return self.data.astype(np.int16) - reference_data.astype(np.int16)

    def get_sub_blocks(self):
        sub_block_size = self.block_size // 2
        for start_row in range(0, self.block_size, sub_block_size):
            for start_col in range(0, self.block_size, sub_block_size):
                yield YuvBlock(
                    self.data[
                        start_row : start_row + sub_block_size,
                        start_col : start_col + sub_block_size,
                    ],
                    sub_block_size,
                    self.row_position + start_row,
                    self.col_position + start_col,
                )

class YuvFrame:
    @staticmethod
    def get_pad_size(height, width, block_size):
        pad_height = (
            block_size - (height % block_size)
            if height % block_size != 0
            else 0
        )
        pad_width = (
            block_size - (width % block_size)
            if width % block_size != 0
            else 0
        )
        return pad_height, pad_width
    
    @staticmethod
    def get_padded_size(height, width, block_size):
        pad_height, pad_width = YuvFrame.get_pad_size(height, width, block_size)
        return height + pad_height, width + pad_width

    def __init__(self, data, block_size) -> None:
        self.data = data
        self.shape = data.shape
        self.block_size = block_size
        pad_height, pad_width = YuvFrame.get_pad_size(*self.shape, block_size)
        self.padded_frame = np.pad(
            self.data,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=128,
        )
        self.height, self.width = self.padded_frame.shape
        
    def get_blocks(self) -> YuvBlock:
        for start_row in range(0, self.height, self.block_size):
            for start_col in range(0, self.width, self.block_size):
                yield YuvBlock(
                    self.padded_frame[
                        start_row : start_row + self.block_size,
                        start_col : start_col + self.block_size,
                    ],
                    self.block_size,
                    start_row,
                    start_col,
                )

    def get_block(self, row, col):
        return self.padded_frame[
            row : row + self.block_size,
            col : col + self.block_size,
        ]

    def get_psnr(self, reference_frame):
        return cv2.PSNR(self.data, reference_frame.data)

    def PSNR(self, compressed):
        img1 = self.data.astype(np.float64)
        img2 = compressed.data.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        PIXEL_MAX = 255.0
        if mse == 0:
            return 100
        return 20 * log10(PIXEL_MAX / sqrt(mse))
    
    def display(self, duration=1):
        cv2.imshow("y frame", self.data)
        cv2.waitKey(duration)
