from PixelPerfect.Decoder import VideoDecoder
from PixelPerfect.Encoder import VideoEncoder, CodecConfig
from PixelPerfect.FileIO import get_media_file_path, dump, load, clean_data, read_frames
import matplotlib.pyplot as plt

videos = {
    "garden": ("garden.yuv", 240, 352),
    "foreman": ("foreman_cif-1.yuv", 288, 352),
}


def e3_test():
    filename, height, width = videos["garden"]
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
    encoder = VideoEncoder(height, width, config)
    decoder = VideoDecoder(height, width, config)
    for frame in read_frames(get_media_file_path(filename), height, width, config):
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()




def e4_simple_test():
    filename, height, width = videos["foreman"]
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=4,
        quant_level=2,
        approximated_residual_n=3,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=False,
        FMEEnable=False,
        FastME=True,
    )
    encoder = VideoEncoder(height, width, config)
    decoder = VideoDecoder(height, width, config)
    for frame in read_frames(get_media_file_path(filename), height, width, config):
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()


def a2_FME_test():
    filename, height, width = videos["foreman"]
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=4,
        quant_level=2,
        approximated_residual_n=3,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=False,
        FMEEnable=True,
    )
    encoder = VideoEncoder(height, width, config)
    decoder = VideoDecoder(height, width, config)
    for frame in read_frames(get_media_file_path(filename), height, width, config):
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()


def a2_Fast_test():
    filename, height, width = videos["foreman"]
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=4,
        quant_level=4,
        approximated_residual_n=3,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=False,
        FMEEnable=False,
        FastME=False,
    )
    encoder = VideoEncoder(height, width, config)
    decoder = VideoDecoder(height, width, config)
    for frame in read_frames(get_media_file_path(filename), height, width, config):
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()

def e4_test():
    filename, height, width = videos["foreman"]
    R_D = []
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        i_Period=1,
        quant_level=0,
        approximated_residual_n=2,
        do_approximated_residual=False,
        do_dct=True,
        do_quantization=True,
        do_entropy=False,
        RD_lambda = 0,
        VBSEnable=True,
        FMEEnable=False,
        FastME=False,
    )
    for i_p in [-1, 0, 4, 10]:
        config.i_Period = i_p
        levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for level in levels:
            config.quant_level = level
            encoder = VideoEncoder(height, width, config)
            decoder = VideoDecoder(height, width, config)
            psnr_sum = 0
            for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                compressed_data = encoder.process(frame)
                decoded_frame = decoder.process(compressed_data)
                decoded_frame.display()
                psnr_sum += decoded_frame.get_psnr(frame)
                if seq == 10:
                    print(psnr_sum)
                    R_D.append((encoder.bitrate, (psnr_sum) / len(levels)))
                    break
        print(R_D)
        x = [R_D[i][0] for i in range(len(levels))]
        y = [R_D[i][1] for i in range(len(levels))]
        R_D = []
        plt.plot(x, y, label="i_period=" + str(i_p), linewidth=0.5)
    plt.legend()
    plt.show()

def run_tests():
    e4_test()
