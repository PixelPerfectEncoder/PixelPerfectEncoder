from PixelPerfect.Yuv import YuvVideo, YuvFrame, YuvBlock
from PixelPerfect.Common import CodecConfig
from PixelPerfect.Decoder import Decoder
from PixelPerfect.ResidualProcessor import ResidualProcessor
from bitstring import BitArray
from bitstring import BitStream
import numpy as np

class Encoder():
    def __init__(self, video : YuvVideo, config : CodecConfig):
        self.video = video
        self.config = config
        self.decoder = Decoder(video.meta, config)
        self.residual_processor = ResidualProcessor(config.block_size)
        
    def is_better_match_block(self, di, dj, block : YuvBlock, min_mae, best_i, best_j) -> bool:
        i = block.row_position + di
        j = block.col_position + dj
        block_size = self.config.block_size
        if (0 <= i <= self.decoder_frame.shape[0] - block_size and
            0 <= j <= self.decoder_frame.shape[1] - block_size):
            reference_block_data = self.decoder_frame.data[i:i + block_size, j:j + block_size]
            mae = block.get_mae(reference_block_data)
            if mae > min_mae:
                return False, None
            if mae < min_mae:
                return True, mae
            if abs(di) + abs(dj) > abs(best_i - block.row_position) + abs(best_j - block.col_position):
                return False, None
            if abs(di) + abs(dj) < abs(best_i - block.row_position) + abs(best_j - block.col_position):
                return True, mae
            if di > best_i - block.row_position:
                return False, None
            if di < best_i - block.row_position:
                return True, mae
            if dj < best_j - block.col_position:
                return True, mae
        return False, None
    
    def find_best_match_block(self, block : YuvBlock) -> YuvBlock:
        min_mae = float('inf')
        best_i, best_j = None, None
        offset = self.config.block_search_offset
        for di in range(-offset, offset + 1):
            for dj in range(-offset, offset + 1):
                is_better_match, mae = self.is_better_match_block(di, dj, block, min_mae, best_i, best_j)
                if is_better_match:
                    min_mae = mae
                    best_i, best_j = block.row_position + di, block.col_position + dj
                            
        block_size = self.config.block_size
        return YuvBlock(
            self.decoder_frame.data[best_i:best_i + block_size, best_j:best_j + block_size],
            block_size,
            best_i,
            best_j
        )
    def intra_horizontal_pred(self,cur_block):
        if cur_block.col_position == 0:
            predicted_block = np.full((128, 128), cur_block.block_size)
        else:
            ref_col = self.cur_frame[cur_block.row_position: cur_block.row_position+cur_block.block_size, cur_block.col_position-1:cur_block.col_position]
            predicted_block = ref_col
            while predicted_block.shape[1] < cur_block.block_size:
                np.column_stack((predicted_block,ref_col))
        residual = np.abs(predicted_block - cur_block.data)
        MAE = np.sum(residual)
        return (MAE, residual, predicted_block)

    def intra_vertical_pred(self,cur_block):
        if cur_block.row_position == 0:
            predicted_block = np.full((128, 128), cur_block.block_size)
        else:
            ref_col = self.cur_frame[cur_block.row_position-1: cur_block.row_position,
                      cur_block.col_position:cur_block.col_position+cur_block.block_size]
            predicted_block = np.tile(ref_col, (cur_block.block_size, 1))
        residual = np.abs(predicted_block - cur_block.data)
        MAE = np.sum(residual)
        return (MAE, residual, predicted_block)

    def intra_pred(self,cur_block):
        MAE_h, residual_h, predicted_block_h = self.intra_horizontal_pred(cur_block)
        MAE_v, residual_v, predicted_block_v = self.intra_vertical_pred(cur_block)
        #return mode 0 if horizontal mode has leat MAE
        if MAE_h < MAE_v:
            reconstructed_block = residual_h + predicted_block_h
            return (0, residual_h, reconstructed_block)
        else:
            reconstructed_block = residual_v + predicted_block_v
            return (1, residual_v, reconstructed_block)

    def RLE_coding(self, sequence):
        index = 0
        while index < len(sequence):
            temp = index
            if sequence[temp] == 0:
                indicater = 1
                count = 0
                while temp < len(sequence) and sequence[temp] == 0:
                    count += 1
                    temp += 1
            else:
                indicater = -1
                count = 0
                while temp < len(sequence) and sequence[temp] != 0:
                    count += 1
                    temp += 1
            if indicater == 1:
                if temp >= len(sequence):
                    sequence[index:] = [0]
                else:
                    sequence[index:temp] = [count]
                temp = index
            else:
                sequence.insert(index, indicater * count)
            index = temp + 1
        return sequence

    def entrophy_coding(self, data):
        # extract the element in diagnosed order
        max_col = data.shape[1]
        max_row = data.shape[0]
        fdiag = [[] for _ in range(max_row + max_col - 1)]
        for y in range(max_col):
            for x in range(max_row):
                fdiag[x + y].append(int(data[y][x]))
        sequence = []
        for i in range(len(fdiag)):
            sequence += fdiag[i][:]
        # sequence = [-31, 9, -4, 8, 1, -3, 4, 4, 2, 4, 0, 4, 0, 0, -4, 0]
        #RLE coding
        original = sequence
        sequence = self.RLE_coding(sequence)
        #Exponential-Golomb Coding
        for i in sequence:
            bit_sequence = [BitArray(se=i) for i in sequence]
        return bit_sequence, original

    def process(self):
        self.decoder_frame = YuvFrame(np.full((self.video.meta.height, self.video.meta.width), 128))
        self.decoder = Decoder(self.video.meta, self.config)
        count = 0
        p_frame = True
        for frame in self.video.get_y_frames():
            self.cur_frame = frame
            # if count%4 == 0:
            #     p_frame = False
            # else:
            #     p_frame = True
            compressed_data = []
            if p_frame:
                for block in frame.get_blocks(self.config.block_size):
                    best_match_block = self.find_best_match_block(block)
                    residual = block.get_residual(best_match_block.data)
                    dct_residual = self.residual_processor.dct_transform(residual)
                    quantized_dct = self.residual_processor.quantization(dct_residual)
                    Entrophy_coded_data, sequence =self.entrophy_coding(quantized_dct)
                    if self.config.do_approximated_residual:
                        residual = self.residual_processor.encode(residual)
                    compressed_data.append((
                        best_match_block.row_position,
                        best_match_block.col_position,
                        residual,
                        Entrophy_coded_data,
                        sequence
                        ))
            else:
                for block in frame.get_blocks(self.config.block_size):
                    me, residual, reconstructed_block = self.intra_pred(block)
                    self.cur_frame[block.row_position: block.row_position+block.block_size,
                    block.col_position: block.col_position+block.block_size] = reconstructed_block
                    #reconstruct the frame

            count+=1
            yield compressed_data
            self.decoder_frame = self.decoder.process(compressed_data)
