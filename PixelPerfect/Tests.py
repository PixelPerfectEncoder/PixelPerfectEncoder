from PixelPerfect.Decoder import Decoder
from PixelPerfect.Encoder import Encoder, CodecConfig
from PixelPerfect.Yuv import YuvInfo
from PixelPerfect.FileIO import get_media_file_path, dump, load, clean_data


def play_foreman_test():
    filename = "foreman_cif-1.yuv"
    video_info = YuvInfo(height=288, width=352)
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=-1,
        quant_level=2,
        approximated_residual_n=2,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=True,
    )
    encoder = Encoder(video_info, config, get_media_file_path(filename))
    decoder = Decoder(video_info, config)
    for compressed_data in encoder.process():
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()


def e3_test():
    filename = "foreman_cif-1.yuv"
    video_info = YuvInfo(height=288, width=352)
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
    encoder = Encoder(video_info, config, get_media_file_path(filename))
    decoder = Decoder(video_info, config)
    for compressed_data in encoder.process():
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()


def e4_test():
    filename = "foreman_cif-1.yuv"
    video_info = YuvInfo(height=288, width=352)
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=3,
        quant_level=2,
        approximated_residual_n=2,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=True,
    )
    encoder = Encoder(video_info, config, get_media_file_path(filename))
    decoder = Decoder(video_info, config)
    for compressed_data in encoder.process():
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()


def run_tests():
    e3_test()
    # e4_test()
    # play_foreman_test()
