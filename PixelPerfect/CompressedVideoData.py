from typing import List
from PixelPerfect.CompressedFrameData import CompressedFrameData
from PixelPerfect.CodecConfig import CodecConfig

class CompressedVideoData:
    def __init__(self, config: CodecConfig, stream: List[CompressedFrameData], psnr: float):
        self.bitrate = 0
        for frame in stream:
            self.bitrate += frame.bit_rate
        self.psnr = psnr
        self.config = config
        self.stream = stream