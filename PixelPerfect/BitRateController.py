from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.Yuv import YuvFrame
import bisect

class BitRateController:
    def __init__(self, height: int, width: int, config: CodecConfig) -> None:        
        self.config = config
        if self.config.RCflag == 1:
            self.left_budget = int(self.config.targetBR * 1024 * (self.config.total_frames / self.config.fps))
            self.left_frames = self.config.total_frames
            _, padded_height = YuvFrame.get_padded_size(height, width, self.config.block_size)
            self.block_rows_per_frame = padded_height // self.config.block_size
            self.i_frame_bit_count = [(int(bc), int(qp)) for qp, bc in self.config.RCTable['I'].items()]
            self.p_frame_bit_count = [(int(bc), int(qp)) for qp, bc in self.config.RCTable['P'].items()]
            self.i_frame_bit_count.sort()
            self.p_frame_bit_count.sort()
        
    def use_bit_count_for_a_frame(self, bit_count: int):
        self.left_frames -= 1
        self.left_budget -= bit_count
    
    def _get_budget_per_block_row(self) -> int:
        return int(self.left_budget // (self.block_rows_per_frame * self.left_frames))
    
    def _find_closest_qp(self, budget: int, is_i_frame: int) -> int:
        bit_count = self.i_frame_bit_count if is_i_frame else self.p_frame_bit_count
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
        else:
            raise Exception("Error! RCflag not supported by BitRateController")
    
    