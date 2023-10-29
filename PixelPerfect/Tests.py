from PixelPerfect.Decoder import Decoder
from PixelPerfect.Encoder import Encoder, CodecConfig
from PixelPerfect.Yuv import YuvInfo
from PixelPerfect.FileIO import get_media_file_path, dump, load, clean_data, read_frames


videos = {
    'garden': ('garden.yuv', YuvInfo(height=240, width=352)),
    'foreman': ('foreman_cif-1.yuv', YuvInfo(height=288, width=352)),
}

def e3_test():
    filename, video_info = videos['garden']
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=-1,
        quant_level=2,
        approximated_residual_n=3,
        do_approximated_residual=True,
        do_dct=False,
        do_quantization=False,
        do_entropy=False,
    )
    encoder = Encoder(video_info, config)
    decoder = Decoder(video_info, config)
    for frame in read_frames(get_media_file_path(filename), video_info, config):
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()

def e4_test():
    filename, video_info = videos['foreman']
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=4,
        quant_level=2,
        approximated_residual_n=3,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=True,
    )
    encoder = Encoder(video_info, config)
    decoder = Decoder(video_info, config)
    for frame in read_frames(get_media_file_path(filename), video_info, config):
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()

def run_tests():
    e3_test()
    # e4_test()
    # play_foreman_test()
