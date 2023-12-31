import numpy as np
import cv2
from math import log10, sqrt, isclose
from PixelPerfect.CodecConfig import CodecConfig

class YuvBlock:
    def __init__(self, data: np.ndarray, block_size: int, row: int, col: int) -> None:
        self.data: np.ndarray = data
        self.block_size: int = block_size
        self.row: int = row
        self.col: int = col

    def add_residual(self, residual: np.ndarray):
        self.data = self.data + residual

    def get_mae(self, ref_block) -> float:
        return np.mean(np.abs(self.data.astype(np.int16) - ref_block.data.astype(np.int16)))

    def get_SAD(self, ref_block) -> float:
        return np.sum(np.abs(self.data.astype(np.int16) - ref_block.data.astype(np.int16)))

    def get_residual(self, ref_block) -> np.ndarray:
        return self.data.astype(np.int16) - ref_block.data.astype(np.int16)

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
                    self.row + start_row,
                    self.col + start_col,
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

    def __init__(self, config: CodecConfig, data: np.ndarray) -> None:
        self.data = data
        self.config = config
        self.block_size = self.config.block_size
        pad_height, pad_width = YuvFrame.get_pad_size(*self.data.shape, self.block_size)
        self.data = np.pad(
            self.data,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=128,
        )
        self.height, self.width = self.data.shape
        if self.config.DisplayRefFrames:
            self.frame_seq2color = dict()
            interval = 255 // self.config.nRefFrames
            for i in range(self.config.nRefFrames):
                self.frame_seq2color[i] = i * interval

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
    
    def MAE(self, compressed):
        return np.mean(np.abs(self.data.astype(np.int16) - compressed.data.astype(np.int16)))
    
    def display(self, duration=1):
        cv2.imshow("y frame", self.data)
        cv2.waitKey(duration)

    def draw_mv(self, row, col, row_mv, col_mv, block_size):
        target = (int(col + block_size / 2), int(row + block_size / 2))
        center = (int(col + block_size / 2 + col_mv), int(row + block_size / 2 + row_mv))
        cv2.arrowedLine(self.data, center, target, color=0, thickness=1, tipLength=0.3)

    def draw_mode(self, row, col, block_size, mode):
        if mode == 0: # vertical
            if row - 1 < 0:
                return
            start = (col, row - 1)
            end = (col + block_size, row - 1)
        else: # horizontal
            if col - 1 < 0:
                return
            start = (col - 1, row)
            end = (col - 1, row + block_size)
        cv2.line(self.data, start, end, color=0, thickness=1)

    def draw_ref_frame(self, row, col, block_size, distance):
        color = self.frame_seq2color[distance]
        cv2.rectangle(self.data, (col, row), (col + block_size, row + block_size), color=color, thickness=-1)
    
    def draw_block(self, row, col, block_size):
        cv2.rectangle(self.data, (col, row), (col + block_size, row + block_size), color=0, thickness=1)


class ConstructingFrame(YuvFrame):
    def __init__(self, config: CodecConfig, height, width) -> None:
        super().__init__(config, np.zeros(shape=[height, width], dtype=np.uint8))

    def put_block(self, row, col, block: YuvBlock):
        if not isclose(row % 1, 0) or not isclose(col % 1, 0):
            raise Exception(f"Error! Invalid block position {row}, {col}")
        self.data[
            int(row) : int(row) + block.block_size,
            int(col) : int(col) + block.block_size,
        ] = block.data.clip(0, 255)

    def get_block(self, row, col, is_sub_block) -> YuvBlock:
        block_size = self.config.sub_block_size if is_sub_block else self.block_size
        return YuvBlock(
            self.data[
                row : row + block_size,
                col : col + block_size,
            ],
            block_size,
            row,
            col,
        )
        
    def get_vertical_ref_block(self, row, col, is_sub_block: bool) -> YuvBlock:
        block_size = self.config.sub_block_size if is_sub_block else self.block_size
        if row == 0:
            return YuvBlock(
                np.zeros((block_size, block_size), dtype=np.uint8),
                block_size,
                row,
                col,
            )  
        vertical_ref_row = self.data[
            row - 1 : row,
            col : col + block_size,
        ]
        return YuvBlock(
            np.repeat(vertical_ref_row, repeats=block_size, axis=0),
            block_size,
            row,
            col,
        )
        
    def get_horizontal_ref_block(self, row, col, is_sub_block: bool) -> YuvBlock:
        block_size = self.config.sub_block_size if is_sub_block else self.block_size
        if col == 0:
            return YuvBlock(
                np.zeros((block_size, block_size), dtype=np.uint8),
                block_size,
                row,
                col,
            )
        horizontal_ref_col = self.data[
            row : row + block_size,
            col - 1 : col,
        ]
        return YuvBlock(
            np.repeat(horizontal_ref_col, repeats=block_size, axis=1),
            block_size,
            row,
            col,
        )
        
    def to_reference_frame(self):
        return ReferenceFrame(self.config, self.data)

class ReferenceFrame(YuvFrame):
    def __init__(self, config: CodecConfig, data: np.ndarray) -> None:
        super().__init__(config, data)
        if self.config.FMEEnable:
            self.create_FME_ref()
        self.cross_area_moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    def create_FME_ref(self):
        x, y = self.data.shape
        self.FME_frame = np.zeros((2 * x - 1, 2 * y - 1), dtype=np.uint8)
        for i in range(x):
            for j in range(y):
                self.FME_frame[2 * i, 2 * j] = self.data[i, j]
                # Calculate the average of neighbors and store it in the result array
                if i < x - 1:
                    self.FME_frame[2 * i + 1, 2 * j] = round(
                        self.data[i, j] / 2
                        + self.data[i + 1, j] / 2
                    )
                if j < y - 1:
                    self.FME_frame[2 * i, 2 * j + 1] = round(
                        self.data[i, j] / 2
                        + self.data[i, j + 1] / 2
                    )
                if i < x - 1 and j < y - 1:
                    self.FME_frame[2 * i + 1, 2 * j + 1] = round(
                        self.data[i + 1, j + 1] / 2
                        + self.data[i, j] / 2
                    )

    def get_ref_blocks_in_cross_area(self, center_block: YuvBlock) -> YuvBlock:
        block_size = center_block.block_size
        row = center_block.row
        col = center_block.col
        if self.config.FMEEnable:
            for drow, dcol in self.cross_area_moves:
                r = int(row * 2 + drow)
                c = int(col * 2 + dcol)
                if r < 0 or r + block_size * 2 > self.FME_frame.shape[0] or c < 0 or c + block_size * 2 > self.FME_frame.shape[1]:
                    continue
                yield YuvBlock(
                    self.FME_frame[
                        r : r + block_size * 2 : 2,
                        c : c + block_size * 2 : 2,
                    ],
                    block_size,
                    r / 2,
                    c / 2,
                )
        else:   
            for drow, dcol in self.cross_area_moves:
                r = row + drow
                c = col + dcol
                if r < 0 or r + block_size > self.height or c < 0 or c + block_size > self.width:
                    continue
                yield YuvBlock(
                    self.data[
                        r : r + block_size,
                        c : c + block_size,
                    ],
                    block_size,
                    r,
                    c,
                )
    
    def get_ref_blocks_in_offset_area(self, center_block: YuvBlock) -> YuvBlock:
        block_size = center_block.block_size
        row = center_block.row
        col = center_block.col
        offset = self.config.block_search_offset
        if self.config.FMEEnable:
            row = int(row * 2)
            col = int(col * 2)
            offset *= 2
            row_start = max(0, row - offset)
            row_end = min(self.FME_frame.shape[0] - block_size * 2, row + offset)
            col_start = max(0, col - 2 * offset)
            col_end = min(self.FME_frame.shape[1] - block_size * 2, col + offset)
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1): 
                    yield YuvBlock(
                        self.FME_frame[
                            r : r + block_size * 2 : 2,
                            c : c + block_size * 2 : 2,
                        ],
                        block_size,
                        r / 2,
                        c / 2,
                    )
        else:
            row_start = max(0, row - offset)
            row_end = min(self.height - block_size, row + offset)
            col_start = max(0, col - offset)
            col_end = min(self.width - block_size, col + offset)
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    yield YuvBlock(
                        self.data[
                            r : r + block_size,
                            c : c + block_size,
                        ],
                        block_size,
                        r,
                        c,
                    )
        
    def get_blocks(self) -> YuvBlock:
        for start_row in range(0, self.height, self.block_size):
            for start_col in range(0, self.width, self.block_size):
                yield YuvBlock(
                    self.data[
                        start_row : start_row + self.block_size,
                        start_col : start_col + self.block_size,
                    ],
                    self.block_size,
                    start_row,
                    start_col,
                )

    def get_block(self, row, col, is_sub_block) -> YuvBlock:
        block_size = self.config.sub_block_size if is_sub_block else self.block_size
        if self.config.FMEEnable:
            data = self.FME_frame[
                int(row * 2) : int(row * 2) + block_size * 2 : 2,
                int(col * 2) : int(col * 2) + block_size * 2 : 2,
            ]
        else:
            data = self.data[
                row : row + block_size,
                col : col + block_size,
            ]
        return YuvBlock(
                data,
                block_size,
                row,
                col,
            )

    def get_block_by_mv(self, row, col, row_mv, col_mv, block_size: int) -> YuvBlock:
        if self.config.FMEEnable:
            fme_row = int(row * 2 + row_mv * 2)
            fme_col = int(col * 2 + col_mv * 2)
            data = self.FME_frame[
                fme_row : fme_row + block_size * 2 : 2,
                fme_col : fme_col + block_size * 2 : 2,
            ]
        else:
            data = self.data[
                row + row_mv : row + row_mv + block_size,
                col + col_mv : col + col_mv + block_size,
            ]
        return YuvBlock(
            data,
            block_size,
            row + row_mv,
            col + col_mv,
        )