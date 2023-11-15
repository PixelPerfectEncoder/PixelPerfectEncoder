import numpy as np
from PixelPerfect.Yuv import YuvBlock, YuvFrame
from PixelPerfect.Coder import Coder, CodecConfig, VideoCoder
from PixelPerfect.Decoder import VideoDecoder, IntraFrameDecoder, InterFrameDecoder


class InterFrameEncoder(Coder):
    def __init__(self, height, width, previous_frame, config: CodecConfig):
        super().__init__(height, width, config)
        self.previous_frame = previous_frame
        self.inter_decoder = InterFrameDecoder(height, width, previous_frame, config)
        if self.config.FMEEnable:
            self.create_FME_ref(self.previous_frame)

    def is_better_match_block(
        self, di, dj, block: YuvBlock, min_mae, best_i, best_j
    ) -> bool:
        i = block.row_position + di
        j = block.col_position + dj
        block_size = block.block_size
        if (
            0 <= i <= self.padded_height - block_size
            and 0 <= j <= self.padded_width - block_size
        ):
            reference_block_data = self.previous_frame.data[
                i : i + block_size, j : j + block_size
            ]
            mae = block.get_mae(reference_block_data)
            if mae > min_mae:
                return False, None
            if mae < min_mae:
                return True, mae
            # if mae is equal, we need to compare the distance
            if abs(di) + abs(dj) > abs(best_i - block.row_position) + abs(best_j - block.col_position):
                return False, None
            if abs(di) + abs(dj) < abs(best_i - block.row_position) + abs(best_j - block.col_position):
                return True, mae
            # if distance is equal, we prefer the smaller di
            if di > best_i - block.row_position:
                return False, None
            if di < best_i - block.row_position:
                return True, mae
            # if di is equal, we prefer the smaller dj
            if dj < best_j - block.col_position:
                return True, mae
        return False, None

    def is_better_FME_match_block(
        self, di, dj, block: YuvBlock, min_mae, best_i, best_j
    ) -> bool:
        i = block.row_position * 2 + di
        j = block.col_position * 2 + dj
        block_size = block.block_size
        height, width = self.FME_ref_frame.shape
        if (
            0 <= i <= height - block_size * 2 + 1
            and 0 <= j <= width - block_size * 2 + 1
        ):
            reference_block_data = self.FME_ref_frame[
                i : i + (block_size * 2) : 2, j : j + (block_size * 2) : 2
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

        # Calculate the average and store it

    def get_inter_data_fast_search(self, block: YuvBlock, mv_row_pred, mv_col_pred):
        block_size = block.block_size
        if self.config.FMEEnable:
            block_size = block_size * 2
            height_bound = self.height * 2
            width_bound = self.width * 2
            row = min(max(0, block.row_position*2 + mv_row_pred), height_bound - block_size)
            col = min(max(0, block.col_position*2 + mv_col_pred), width_bound - block_size)
            step = 2
            ref_frame = self.FME_ref_frame
        else:
            height_bound = self.height
            width_bound = self.width
            row = min(max(0, block.row_position + mv_row_pred), height_bound - block_size)
            col = min(max(0, block.col_position + mv_col_pred), width_bound - block_size)
            ref_frame = self.previous_frame
            step =1
        mae = block.get_mae(
            ref_frame.data[
            row: row + block_size: step,
            col: col + block_size: step,
            ]
        )
        search_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        while True:
            updated = False
            for drow, dcol in search_moves:
                new_row = row + drow
                new_col = col + dcol
                if (
                    new_row < 0
                    or new_row + block_size > height_bound
                    or new_col < 0
                    or new_col + block_size > width_bound
                ):
                    continue
                new_mae = block.get_mae(
                    ref_frame.data[
                    new_row: new_row + block_size: step,
                    new_col: new_col + block_size: step,
                    ]
                )
                if new_mae < mae:
                    mae = new_mae
                    row, col = new_row, new_col
                    updated = True
            if not updated:
                break
        if self.config.FMEEnable:
            best_di, best_dj = row - block.row_position * 2, col - block.col_position * 2
        else:
            best_di, best_dj = row - block.row_position, col - block.col_position
        return block.get_residual(ref_frame.data[row : row + block_size: step,col : col + block_size: step]), best_di, best_dj

    def get_inter_data_normal_search(self, block: YuvBlock):
        block_size = block.block_size
        min_mae = float("inf")
        best_i, best_j = None, None
        best_di, best_dj = None, None
        if self.config.FMEEnable:
            offset = self.config.block_search_offset * 2
            for di in range(-offset, offset + 1):
                for dj in range(-offset, offset + 1):
                    is_better_match, mae = self.is_better_FME_match_block(
                        di, dj, block, min_mae, best_i, best_j
                    )
                    if is_better_match:
                        min_mae = mae
                        best_i, best_j = (
                            block.row_position * 2 + di,
                            block.col_position * 2 + dj,
                        )
                        best_di, best_dj = di, dj
            reference_block_data = self.FME_ref_frame[
                best_i : best_i + (block_size * 2) : 2,
                best_j : best_j + (block_size * 2) : 2,
            ]
            return block.get_residual(reference_block_data), best_di, best_dj
        else:
            offset = self.config.block_search_offset
            for di in range(-offset, offset + 1):
                for dj in range(-offset, offset + 1):
                    is_better_match, mae = self.is_better_match_block(
                        di, dj, block, min_mae, best_i, best_j
                    )
                    if is_better_match:
                        min_mae = mae
                        best_i, best_j = (
                            block.row_position + di,
                            block.col_position + dj,
                        )
                        best_di, best_dj = di, dj
        return (
            block.get_residual(
                self.previous_frame.data[
                    best_i : best_i + block_size, best_j : best_j + block_size
                ]
            ),
            best_di,
            best_dj,
        )

    def get_inter_data(self, block: YuvBlock, last_row_mv, last_col_mv):
        if self.config.FastME:
            return self.get_inter_data_fast_search(block, last_row_mv, last_col_mv)
        else:
            return self.get_inter_data_normal_search(block)

    # this function should be idempotent
    def process(self, block: YuvBlock, block_seq: int, last_row_mv: int, last_col_mv: int, use_sub_blocks: bool):
        compressed_residual = []
        descriptors = []
        residual_bitrate = 0
        if use_sub_blocks:
            for sub_block_seq, sub_block in enumerate(block.get_sub_blocks()):
                residual, row_mv, col_mv = self.get_inter_data(sub_block, last_row_mv, last_col_mv)
                residual, bitrate = self.compress_residual(residual)
                compressed_residual.append(residual)
                residual_bitrate += bitrate
                descriptors.append(row_mv - last_row_mv)
                descriptors.append(col_mv - last_col_mv)
                descriptors.append(1)
                last_row_mv, last_col_mv = row_mv, col_mv
                self.inter_decoder.process(block_seq, sub_block_seq, residual, row_mv, col_mv, is_sub_block=True)    
        else:
            residual, row_mv, col_mv = self.get_inter_data(block, last_row_mv, last_col_mv)
            residual, bitrate = self.compress_residual(residual)
            residual_bitrate += bitrate
            use_sub_blocks = False
            compressed_residual.append(residual)
            descriptors.append(row_mv - last_row_mv)
            descriptors.append(col_mv - last_col_mv)
            if self.config.VBSEnable:
                descriptors.append(0)
            last_row_mv, last_col_mv = row_mv, col_mv
            self.inter_decoder.process(block_seq, 0, residual, row_mv, col_mv, is_sub_block=False)
        reconstructed_block = self.inter_decoder.frame[
            block.row_position : block.row_position + block.block_size,
            block.col_position : block.col_position + block.block_size,
        ]
        distortion = block.get_SAD(reconstructed_block)
        return compressed_residual, descriptors, distortion, residual_bitrate, last_row_mv, last_col_mv


class IntraFrameEncoder(Coder):
    def __init__(self, height, width, config: CodecConfig, current_frame):
        super().__init__(height, width, config)
        self.frame = current_frame
        self.intra_decoder = IntraFrameDecoder(height, width, config)
            
    def get_intra_data(self, block: YuvBlock):
        vertical_ref = np.full([block.block_size, block.block_size], 128, dtype=np.uint8)
        if block.row_position != 0:
            vertical_ref_row = self.intra_decoder.frame[
                block.row_position - 1 : block.row_position,
                block.col_position : block.col_position + block.block_size,
            ]
            vertical_ref = np.repeat(
                vertical_ref_row, repeats=block.block_size, axis=0
            )
        horizontal_ref = np.full([block.block_size, block.block_size], 128, dtype=np.uint8)
        if block.col_position != 0:
            horizontal_ref_col = self.intra_decoder.frame[
                block.row_position : block.row_position + block.block_size,
                block.col_position - 1 : block.col_position,
            ]
            horizontal_ref = np.repeat(
                horizontal_ref_col, repeats=block.block_size, axis=1
            )
        vertical_mae = block.get_mae(vertical_ref)
        horizontal_mae = block.get_mae(horizontal_ref)
        if vertical_mae < horizontal_mae:
            return block.get_residual(vertical_ref), 0
        else:
            return block.get_residual(horizontal_ref), 1

    # this function should be idempotent
    def process(self, block: YuvBlock, block_seq: int, use_sub_blocks: bool):
        compressed_residual = []
        descriptors = []
        residual_bitrate = 0
        if use_sub_blocks:
            for sub_block_seq, sub_block in enumerate(block.get_sub_blocks()):
                residual, mode = self.get_intra_data(sub_block)
                residual, bitrate = self.compress_residual(residual)
                residual_bitrate += bitrate
                self.intra_decoder.process(block_seq, sub_block_seq, residual, mode, True)
                compressed_residual.append(residual)
                descriptors.append(mode)
                descriptors.append(1)
        else:
            residual, mode = self.get_intra_data(block)
            residual, bitrate = self.compress_residual(residual)
            residual_bitrate += bitrate
            self.intra_decoder.process(block_seq, 0, residual, mode, False)
            compressed_residual.append(residual)
            descriptors.append(mode)
            if self.config.VBSEnable:
                descriptors.append(0)
            
        reconstructed_block = self.intra_decoder.frame[
            block.row_position : block.row_position + block.block_size,
            block.col_position : block.col_position + block.block_size,
        ]
        distortion = block.get_SAD(reconstructed_block)
        return compressed_residual, descriptors, distortion, residual_bitrate


class VideoEncoder(VideoCoder):
    def __init__(self, height, width, config: CodecConfig):
        super().__init__(height, width, config)

    def calculate_RDO(self, bitrate, distortion):
        return distortion + self.config.RD_lambda * bitrate

    def process_p_frame(self, frame: YuvFrame):
        compressed_residual = []
        descriptors = []
        last_row_mv, last_col_mv = 0, 0
        frame_encoder = InterFrameEncoder(self.height, self.width, self.previous_frame, self.config)
        for block_seq, block in enumerate(frame.get_blocks()):
            (
                normal_residual,
                normal_descriptors,
                normal_distortion,
                normal_residual_bitrate,
                normal_last_row_mv,
                normal_last_col_mv,
            ) = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=False)
            normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
            use_sub_blocks = False
            if self.config.VBSEnable:
                (
                    sub_blocks_residual,
                    sub_blocks_descriptors,
                    sub_blocks_distortion,
                    sub_blocks_residual_bitrate,
                    sub_blocks_last_row_mv,
                    sub_blocks_last_col_mv,
                ) = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=True)
                sub_blocks_bitrate = sub_blocks_residual_bitrate + len(sub_blocks_descriptors)
                use_sub_blocks = self.calculate_RDO(normal_bitrate, normal_distortion) > self.calculate_RDO(sub_blocks_bitrate, sub_blocks_distortion)
                # roll back normal block status
                if not use_sub_blocks:
                    _ = frame_encoder.process(block, block_seq, last_row_mv, last_col_mv, use_sub_blocks=False)

            if use_sub_blocks:
                compressed_residual += sub_blocks_residual
                descriptors += sub_blocks_descriptors
                self.bitrate += sub_blocks_residual_bitrate
                last_row_mv, last_col_mv = sub_blocks_last_row_mv, sub_blocks_last_col_mv
            else:
                compressed_residual += normal_residual
                descriptors += normal_descriptors
                self.bitrate += normal_residual_bitrate
                last_row_mv, last_col_mv = normal_last_row_mv, normal_last_col_mv

        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        self.bitrate += descriptors_bitrate
        compressed_data = (compressed_residual, compressed_descriptors)
        decoded_frame = YuvFrame(frame_encoder.inter_decoder.frame, self.config.block_size)
        self.frame_processed(decoded_frame)
        return compressed_data

    def process_i_frame(self, frame: YuvFrame):
        compressed_residual = []
        descriptors = []
        frame_encoder = IntraFrameEncoder(self.height, self.width, self.config, frame.data)
        for block_seq, block in enumerate(frame.get_blocks()):
            (
                normal_residual,
                normal_descriptors,
                normal_distortion,
                normal_residual_bitrate,
            ) = frame_encoder.process(block, block_seq, use_sub_blocks=False)
            normal_bitrate = normal_residual_bitrate + len(normal_descriptors)
            use_sub_blocks = False
            if self.config.VBSEnable:
                (
                    sub_blocks_residual,
                    sub_blocks_descriptors,
                    sub_blocks_distortion,
                    sub_blocks_residual_bitrate,
                ) = frame_encoder.process(block, block_seq, use_sub_blocks=True)
                sub_blocks_bitrate = sub_blocks_residual_bitrate + len(sub_blocks_descriptors)
                use_sub_blocks = self.calculate_RDO(normal_bitrate, normal_distortion) > self.calculate_RDO(sub_blocks_bitrate, sub_blocks_distortion)
                # roll back normal block status
                if not use_sub_blocks:
                    _ = frame_encoder.process(block, block_seq, use_sub_blocks=False)
            if use_sub_blocks:
                compressed_residual += sub_blocks_residual
                descriptors += sub_blocks_descriptors
                self.bitrate += sub_blocks_residual_bitrate
            else:
                compressed_residual += normal_residual
                descriptors += normal_descriptors
                self.bitrate += normal_residual_bitrate
            
        compressed_descriptors, descriptors_bitrate = self.compress_descriptors(descriptors)
        self.bitrate += descriptors_bitrate
        compressed_data = (compressed_residual, compressed_descriptors)
        decoded_frame = YuvFrame(frame_encoder.intra_decoder.frame, self.config.block_size)
        self.frame_processed(decoded_frame)
        return compressed_data

    def process(self, frame: YuvFrame):
        if self.is_i_frame():
            return self.process_i_frame(frame)
        else:
            return self.process_p_frame(frame)
