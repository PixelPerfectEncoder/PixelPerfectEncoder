from typing import List
from PixelPerfect.CompressedFrameData import CompressedFrameData
from PixelPerfect.CodecConfig import CodecConfig

class StreamMetrics:
    def __init__(self, psnr, bitrate):
        self.psnr = psnr
        self.bitrate = bitrate

class CompressedVideoData:
    def __init__(self, config: CodecConfig, stream: List[CompressedFrameData], metrics: StreamMetrics):
        self.config = config
        self.stream = stream
        self.metrics = metrics