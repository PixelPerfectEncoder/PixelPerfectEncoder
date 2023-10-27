from PixelPerfect.Decoder import Decoder
from PixelPerfect.Encoder import Encoder, CodecConfig
from PixelPerfect.Yuv import YuvVideo, YuvMeta
from PixelPerfect.FileIO import get_media_file_path, dump, load, clean_data

def play_foreman_test():
    filename = 'foreman_cif-1.yuv'
    video_info = YuvMeta(height=288, width=352)
    video = YuvVideo(get_media_file_path(filename), video_info)
    config = CodecConfig(block_size=16, block_search_offset=2)
    encoder = Encoder(video, config)
    decoder = Decoder(video_info, config)
    file_ids = []
    for compressed_data in encoder.process():
        file_id = dump(compressed_data)
        file_ids.append(file_id)
        decoded_frame = decoder.process(load(file_id))
        decoded_frame.display()
    clean_data(file_ids)

def play_foreman_test_better_quality():
    filename = 'foreman_cif-1.yuv'
    video_info = YuvMeta(height=288, width=352)
    video = YuvVideo(get_media_file_path(filename), video_info)
    config = CodecConfig(
        block_size=16, 
        block_search_offset=2, 
        do_approximated_residual=False,
        do_dct=False,
        do_quantization=False,
        do_entropy=False,
    )
    encoder = Encoder(video, config)
    decoder = Decoder(video_info, config)
    encoded = []
    for compressed_data in encoder.process():
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()

def run_tests():
    play_foreman_test_better_quality()

