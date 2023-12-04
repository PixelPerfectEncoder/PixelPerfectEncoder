from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.Yuv import YuvFrame
import bisect

class BitRateController:
    def __init__(self, height: int, width: int, config: CodecConfig) -> None:        
        self.config = config
        if self.config.RCflag == 1:
            self.budget_per_frame = int(self.config.targetBR * 1024 * (self.config.total_frames / self.config.fps)/self.config.total_frames)
            padded_height, _ = YuvFrame.get_padded_size(height, width, self.config.block_size)
            self.left_budget = self.budget_per_frame
            self.block_rows_per_frame = padded_height // self.config.block_size
            self.coded_rows = 0
            self.i_frame_bit_count_sorted = [(int(bc), int(qp)) for qp, bc in self.config.RCTable['I'].items()]
            self.p_frame_bit_count_sorted = [(int(bc), int(qp)) for qp, bc in self.config.RCTable['P'].items()]
            self.p_frame_bit_count_sorted.sort()
            self.i_frame_bit_count_sorted.sort()
        if self.config.RCflag > 1:
            if self.config.filename =='QCIF':
                self.threshold_dic = \
                    {0: 116571.0, 1: 118777.0, 2: 91471.0, 3: 67221.5, 4: 48519.0, 5: 33851.5, 6: 23138.5, 7: 16429.5, 8: 13960.0, 9: 10463.5, 10: 9133.0, 11: 8906.0}
            elif self.config.filename == 'CIF':
                self.threshold_dic = \
                    {0: 455288.5, 1: 460389.0, 2: 346074.0, 3: 243604.0, 4: 162351.0, 5: 103956.0, 6: 69749.0, 7: 53619.0, 8: 45148.0, 9: 38459.0, 10: 34059.0, 11: 33381.0}
            self.left_budget = int(self.config.targetBR * 1024 * (self.config.total_frames / self.config.fps))
            self.budget_per_frame = int(
                self.config.targetBR * 1024 * (self.config.total_frames / self.config.fps) / self.config.total_frames)
            padded_height, padded_width = YuvFrame.get_padded_size(height, width, self.config.block_size)
            self.left_budget = self.budget_per_frame
            self.block_rows_per_frame = padded_height // self.config.block_size
            self.block_per_row = padded_width//self.config.block_size
            self.coded_rows = 0
            self.i_frame_bit_count = [(int(bc), int(qp)) for qp, bc in self.config.RCTable['I'].items()]
            self.p_frame_bit_count = [(int(bc), int(qp)) for qp, bc in self.config.RCTable['P'].items()]
            self.i_frame_bit_count_sorted = self.i_frame_bit_count[:]
            self.p_frame_bit_count_sorted = self.p_frame_bit_count[:]
            self.p_frame_bit_count_sorted.sort()
            self.i_frame_bit_count_sorted.sort()
        
    def use_bit_count_for_a_frame(self, bit_count: int):
        if self.config.RCflag == 0:
            return
        self.left_budget -= bit_count

    def update_used_rows(self):
        if self.config.RCflag == 0:
            return
        self.coded_rows+=1

    def refresh_frame(self):
        if self.config.RCflag == 0:
            return
        self.left_budget = self.budget_per_frame
        self.coded_rows = 0

    def update_itable(self, qp, bit_count):
        factor = bit_count / self.i_frame_bit_count[qp][0] / self.block_rows_per_frame
        for i in range(len(self.i_frame_bit_count_sorted)):
            li = list(self.i_frame_bit_count_sorted[i])
            li[0] = self.i_frame_bit_count[li[1]][0]*factor
            self.i_frame_bit_count_sorted[i] = tuple(li)
    def update_ptable(self, qp, bit_count):
        factor = bit_count / self.i_frame_bit_count[qp][0] / self.block_rows_per_frame
        for i in range(len(self.p_frame_bit_count_sorted)):
            li = list(self.p_frame_bit_count_sorted[i])
            li[0] = self.p_frame_bit_count[li[1]][0]*factor
            self.p_frame_bit_count_sorted[i] = tuple(li)

    def set_row_ratio(self, per_row_bit):
        self.per_row_ratio = per_row_bit
    def _get_budget_per_block_row(self) -> int:
        if self.config.RCflag == 1:
            return int(self.left_budget // (self.block_rows_per_frame - self.coded_rows) )
        if self.config.RCflag >1:
            return int(self.left_budget * self.per_row_ratio[self.coded_rows] / sum(self.per_row_ratio[self.coded_rows:]))
    
    def _find_closest_qp(self, budget: int, is_i_frame: int) -> int:
        bit_count = self.i_frame_bit_count_sorted if is_i_frame else self.p_frame_bit_count_sorted
        index = bisect.bisect_left(bit_count, (budget, 0))
        if index == 0:
            return bit_count[0][1]
        if index == len(bit_count):
            return bit_count[-1][1]
        before, before_qp = bit_count[index - 1]
        after, after_qp = bit_count[index]
        if after - budget <= budget - before:
            return after_qp
        else:
            return before_qp
    
    def get_qp(self, is_i_frame) -> int:
        if self.config.RCflag == 0:
            return self.config.qp
        elif self.config.RCflag == 1:
            return self._find_closest_qp(self._get_budget_per_block_row(), is_i_frame)
        elif self.config.RCflag > 1:
            return self._find_closest_qp(self._get_budget_per_block_row(), is_i_frame)
        else:
            raise Exception("Error! RCflag not supported by BitRateController")
    
    