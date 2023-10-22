import os
from PixelPerfect.Decoder import Decoder
from PixelPerfect.Encoder import Encoder, EncoderConfig
from PixelPerfect.Yuv import YuvVideo, YuvMeta

def get_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join('media', filename))

def play_foreman_test():
    filename = 'foreman_cif-1.yuv'
    video_info = YuvMeta(height=288, width=352)
    video = YuvVideo(get_file_path(filename), video_info)
    config = EncoderConfig(block_size=16, block_search_offset=2)
    encoder = Encoder(video, config)
    decoder = Decoder(video_info, config)
    for compressed_data in encoder.process():
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()
        
def run_tests():
    play_foreman_test()