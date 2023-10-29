import numpy as np
import cv2


class YuvInfo:
    def __init__(self, height, width) -> None:
        self.height = height
        self.width = width


class YuvBlock:
    def __init__(self, data, block_size, row, col) -> None:
        self.data = data
        self.block_size = block_size
        self.row_position = row
        self.col_position = col

    def get_mae(self, reference_data: np.ndarray) -> float:
        return np.mean(np.abs(self.data - reference_data))

    def get_residual(self, reference_data: np.ndarray) -> np.ndarray:
        return self.data - reference_data

class YuvFrame:
    def __init__(self, data, block_size) -> None:
        self.data = data
        self.shape = data.shape
        self.block_size = block_size
        pad_height = (
            block_size - (self.shape[0] % block_size)
            if self.shape[0] % block_size != 0
            else 0
        )
        pad_width = (
            block_size - (self.shape[1] % block_size)
            if self.shape[1] % block_size != 0
            else 0
        )
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

    def display(self, duration=1):
        cv2.imshow("y frame", self.data)
        cv2.waitKey(duration)
