import numpy as np
import cv2

class YuvMeta:
    def __init__(self, height, width) -> None:
        self.height = height
        self.width = width
        
class YuvBlock:
    def __init__(self, data, block_size, row, col) -> None:
        self.data = data
        self.block_size = block_size
        self.row_position = row
        self.col_position = col
        
    def get_mae(self, reference_data : np.ndarray) -> float:
        return np.mean(np.abs(self.data - reference_data))
        
    def get_residual(self, reference_data : np.ndarray) -> np.ndarray:
        return self.data - reference_data

    def add_residual(self, residual : np.ndarray) -> np.ndarray:
        return np.clip(self.data + residual, 0, 255)

class YuvFrame:
    def __init__(self, data) -> None:
        self.data = data
        self.shape = data.shape
        
    def get_blocks(self, block_size) -> YuvBlock:
        pad_height = block_size - (self.shape[0] % block_size) if self.shape[0] % block_size != 0 else 0
        pad_width = block_size - (self.shape[1] % block_size) if self.shape[1] % block_size != 0 else 0
        padded_frame = np.pad(self.data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=128)
        padded_shape = padded_frame.shape
        for start_row in range(0, padded_shape[0], block_size):
            for start_col in range(0, padded_shape[1], block_size):
                yield YuvBlock(
                    padded_frame[start_row:start_row + block_size, start_col:start_col + block_size], 
                    block_size,
                    start_row,
                    start_col
                )

    def display(self, duration = 1):
        cv2.imshow('y frame', self.data)
        cv2.waitKey(duration)

        
class YuvVideo:
    def __init__(self, file_path : str, meta : YuvMeta) -> None:
        self.file_path = file_path
        self.meta = meta
        height = self.meta.height
        width = self.meta.width
        self.yuv_frame_size = width * height + (width // 2) * (height // 2) * 2
        self.y_frame_size = width * height
    
    def get_y_frames(self) -> YuvFrame:
        with open(self.file_path, 'rb') as file:
            while True:
                yuv_frame_data = file.read(self.yuv_frame_size)            
                if len(yuv_frame_data) < self.yuv_frame_size:
                    break
                yield YuvFrame(np.frombuffer(yuv_frame_data[:self.y_frame_size], dtype=np.uint8).reshape((self.meta.height, self.meta.width)))


    