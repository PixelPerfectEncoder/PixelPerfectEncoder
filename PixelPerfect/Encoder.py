import numpy as np
from bitstring import BitArray, BitStream
from PixelPerfect.Yuv import YuvInfo, YuvBlock, YuvFrame
from PixelPerfect.Coder import Coder, CodecConfig
from PixelPerfect.Decoder import Decoder
from PixelPerfect.FileIO import read_video

class Encoder(Coder):
    def __init__(self, video_info: YuvInfo, config: CodecConfig, source_path: str):
        super().__init__(video_info, config)
        self.decoder = Decoder(video_info, config)
        self.source_path = source_path

    def read_frames(self):
        height = self.video_info.height
        width = self.video_info.width
        yuv_frame_size = width * height + (width // 2) * (height // 2) * 2
        y_frame_size = width * height
        
        for yuv_frame_data in read_video(self.source_path, yuv_frame_size):
            yield YuvFrame(
                np.frombuffer(yuv_frame_data[:y_frame_size], dtype=np.uint8).reshape(
                    (height, width)
                ),
                self.config.block_size,
            )

    def is_better_match_block(
        self, di, dj, block: YuvBlock, min_mae, best_i, best_j
    ) -> bool:
        i = block.row_position + di
        j = block.col_position + dj
        block_size = block.block_size
        if (
            0 <= i <= self.video_info.height - block_size
            and 0 <= j <= self.video_info.width - block_size
        ):
            reference_block_data = self.previous_frame.data[
                i : i + block_size, j : j + block_size
            ]
            mae = block.get_mae(reference_block_data)
            if mae > min_mae:
                return False, None
            if mae < min_mae:
                return True, mae
            if abs(di) + abs(dj) > abs(best_i - block.row_position) + abs(
                best_j - block.col_position
            ):
                return False, None
            if abs(di) + abs(dj) < abs(best_i - block.row_position) + abs(
                best_j - block.col_position
            ):
                return True, mae
            if di > best_i - block.row_position:
                return False, None
            if di < best_i - block.row_position:
                return True, mae
            if dj < best_j - block.col_position:
                return True, mae
        return False, None

    def find_best_match_block(self, block: YuvBlock) -> YuvBlock:
        min_mae = float("inf")
        best_i, best_j = None, None
        offset = self.config.block_search_offset
        for di in range(-offset, offset + 1):
            for dj in range(-offset, offset + 1):
                is_better_match, mae = self.is_better_match_block(
                    di, dj, block, min_mae, best_i, best_j
                )
                if is_better_match:
                    min_mae = mae
                    best_i, best_j = block.row_position + di, block.col_position + dj

        block_size = self.config.block_size
        return YuvBlock(
            self.previous_frame.data[
                best_i : best_i + block_size, best_j : best_j + block_size
            ],
            block_size,
            best_i,
            best_j,
        )

    def get_inter_data(self, block: YuvBlock):
        best_match_block = self.find_best_match_block(block)
        return (
            block.get_residual(best_match_block.data),
            best_match_block.row_position,
            best_match_block.col_position,
        )

    def get_intra_data(self, block: YuvBlock, current_frame: YuvFrame):
        min_mae = float("inf")
        predicted_data = np.zeros([block.block_size, block.block_size], dtype=np.uint8)
        row, col = block.row_position, block.col_position
        try_positions = []
        if block.col_position != 0:
            try_positions.append((block.row_position, block.col_position - block.block_size))
        if block.row_position != 0:
            try_positions.append((block.row_position - block.block_size, block.col_position))
        for ref_row, ref_col in try_positions:
            data = current_frame.get_block(ref_row, ref_col).data
            mae = block.get_mae(data)
            if mae < min_mae:
                min_mae = mae
                predicted_data = data
                row, col = ref_row, ref_col
        return block.get_residual(predicted_data), row, col

    def RLE_coding(self, data):
        sequence = []
        zero_count = 0
        non_zero_count = 0
        for v in reversed(data):
            if v == 0:
                if non_zero_count != 0:
                    sequence.append(non_zero_count)
                    non_zero_count = 0
                zero_count += 1
            else:
                if zero_count != 0:
                    sequence.append(zero_count)
                    zero_count = 0
                non_zero_count -= 1
                sequence.append(v)
        if non_zero_count != 0:
            sequence.append(non_zero_count)
        else:
            sequence.append(zero_count)
        sequence.reverse()
        return sequence

    def get_diagonal_sequence(self, data):
        max_col = data.shape[1]
        max_row = data.shape[0]
        fdiag = [[] for _ in range(max_row + max_col - 1)]
        for y in range(max_col):
            for x in range(max_row):
                fdiag[x + y].append(data[y, x])
        sequence = []
        for diag in fdiag:
            sequence.extend(diag)
        return sequence

    def entrophy_coding(self, data):
        sequence = self.get_diagonal_sequence(data)
        sequence = self.RLE_coding(sequence)
        bit_sequence = BitStream().join([BitArray(se=i) for i in sequence])
        return bit_sequence

    def process(self):
        for frame in self.read_frames():
            compressed_data = []
            for block in frame.get_blocks():
                # get residual
                if self.is_p_frame():
                    residual, row, col = self.get_inter_data(block)
                else:
                    residual, row, col = self.get_intra_data(block, frame)
                # compress residual
                if self.config.do_approximated_residual:
                    residual = self.residual_processor.approx(residual)
                if self.config.do_dct:
                    residual = self.residual_processor.dct_transform(residual)
                if self.config.do_quantization:
                    residual = self.residual_processor.quantization(residual)
                if self.config.do_entropy:
                    residual = self.entrophy_coding(residual)
                # save compressed block
                compressed_data.append((row, col, residual))
            yield compressed_data
            self.frame_processed(self.decoder.process(compressed_data))
            # self.frame_processed(frame)
