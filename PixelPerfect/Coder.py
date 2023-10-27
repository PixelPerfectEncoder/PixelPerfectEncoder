import numpy as np
from PixelPerfect.Yuv import YuvFrame, YuvInfo
from PixelPerfect.ResidualProcessor import ResidualProcessor


class CodecConfig:
    def __init__(
        self,
        block_size,
        block_search_offset,
        i_Period: int = -1,
        approximated_residual_n = 2,
        do_approximated_residual: bool = False,
        do_dct: bool = False,
        do_quantization: bool = False,
        do_entropy: bool = False,
    ) -> None:
        self.block_size = block_size
        self.block_search_offset = block_search_offset
        self.i_Period = i_Period
        self.approximated_residual_n = approximated_residual_n
        self.do_approximated_residual = do_approximated_residual
        self.do_dct = do_dct
        self.do_quantization = do_quantization
        self.do_entropy = do_entropy


class Coder:
    def __init__(self, video_info: YuvInfo, config: CodecConfig) -> None:
        self.frame_seq = 0
        self.config = config
        self.video_info = video_info
        self.previous_frame = YuvFrame(
            np.full((self.video_info.height, self.video_info.width), 128),
            self.config.block_size,
        )
        self.residual_processor = ResidualProcessor(self.config.block_size, self.config.approximated_residual_n)

    def is_p_frame(self):
        if self.config.i_Period == -1:
            return True
        if self.config.i_Period == 0:
            return False
        if self.frame_seq % self.config.i_Period == 0:
            return False
        else:
            return True

    def frame_processed(self, frame):
        self.frame_seq += 1
        self.previous_frame = frame

    def is_i_frame(self):
        return not self.is_p_frame()

    