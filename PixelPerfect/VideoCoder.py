import numpy as np
from PixelPerfect.Yuv import ReferenceFrame
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.Coder import Coder
from typing import Deque

class VideoCoder(Coder):
    def __init__(self, height, width, config: CodecConfig) -> None:
        super().__init__(height, width, config)
        self.frame_seq = 0
        self.previous_frames: Deque[ReferenceFrame] = Deque(maxlen=config.nRefFrames)
        self.previous_frames.append(ReferenceFrame(config, np.full(shape=(self.height, self.width), fill_value=128, dtype=np.uint8)))
        
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
        if self.is_i_frame():
            self.previous_frames.clear()
        self.previous_frames.append(frame)
        
    def is_i_frame(self):
        return not self.is_p_frame()