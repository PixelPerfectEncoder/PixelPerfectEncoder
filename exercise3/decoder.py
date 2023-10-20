from .Yuv import YuvMeta
from .Encoder import EncoderConfig

class Decoder:
    def __init__(self, yuv_info : YuvMeta, encoder_config : EncoderConfig):
        self.yuv_info = yuv_info
        self.encoder_config = encoder_config
        
    def process(data):
        
        