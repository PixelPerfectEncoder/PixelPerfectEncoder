from PixelPerfect.Decoder import VideoDecoder
from PixelPerfect.Encoder import VideoEncoder
from PixelPerfect.CodecConfig import CodecConfig
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
        FMEEnable=True,
        FastME=True,
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
        do_approximated_residual=True,
        do_dct=True,
        do_quantization=True,
        do_entropy=False,
        RD_lambda = 0,
        VBSEnable=True,
        FMEEnable=True,
        FastME=True,
        nRefFrames=3,
        DisplayMvs=True,
        DisplayBlocks=True,
        DisplayRefFrames=True,
    )
    for i_p in [4]:
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

                psnr_sum += frame.PSNR(decoded_frame)
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

def plot_a_RD_to_bitrate_curve(config: CodecConfig, label: str, show_time=False):
    levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R_D = []
    total_time = 0
    filename, height, width = videos["foreman"]
    for level in levels:
        config.quant_level = level
        encoder = VideoEncoder(height, width, config)
        decoder = VideoDecoder(height, width, config)
        psnr_sum = 0
        for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
            t1 = time.time()
            compressed_data = encoder.process(frame)
            total_time += time.time() - t1
            decoded_frame = decoder.process(compressed_data)
            psnr_sum += frame.PSNR(decoded_frame)
            if seq == 10:
                R_D.append((encoder.bitrate, (psnr_sum) / len(levels)))
                break
        print(f"{label}'s {level} is processed")
    x = [R_D[i][0] for i in range(len(levels))]
    y = [R_D[i][1] for i in range(len(levels))]
    if show_time:
        average_time = total_time / len(levels)
        label += f" (average time: {average_time:.2f}s)"
    plt.plot(x, y, label=label, linewidth=0.5)

def lambda_test():
    config = CodecConfig(
        block_size=16,
        block_search_offset=4,
        i_Period=8,
        do_dct=True,
        do_quantization=True,
        VBSEnable=True,
        DisplayBlocks=True,
    )
    RD_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for RD_lambda in RD_lambdas:
        config.RD_lambda = RD_lambda
        plot_a_RD_to_bitrate_curve(config, RD_lambda, "RD_lambda")
    plt.legend()
    plt.show()

def a2_q1_experiment():
    """
    Create RD plots for a fixed set of parameters (block size = 16, search range = 4, I_Period = 8). There
    should be 6 curves drawn: one for the encoder of part1, one for each feature added in this
    assignment (by itself), and one with all four features. Like in Assignment 1, use the first 10 frames
    of Foreman for testing, and build your curves using at least QPs 1, 4, 7 and 10. It is important to
    include execution times as well in the comparisons.
    """
    config = CodecConfig(
        block_size=16,
        block_search_offset=4,
        i_Period=8,
        do_dct=True,
        do_quantization=True,
    )
    plot_a_RD_to_bitrate_curve(config, label="All features off", show_time=True)
    config.VBSEnable = True
    config.RD_lambda = 0.3
    plot_a_RD_to_bitrate_curve(config, label="VBSEnable", show_time=True)
    config.VBSEnable = False
    config.FMEEnable = True
    plot_a_RD_to_bitrate_curve(config, label="FMEEnable", show_time=True)
    config.FMEEnable = False
    config.FastME = True
    plot_a_RD_to_bitrate_curve(config, label="FastME", show_time=True)
    config.FastME = False
    config.nRefFrames = 3
    plot_a_RD_to_bitrate_curve(config, label="nRefFrames = 3", show_time=True)
    config.VBSEnable = True
    config.FMEEnable = True
    config.FastME = True
    plot_a_RD_to_bitrate_curve(config, label="All features on", show_time=True)
    
def a2_q2_experiment():
    """
    While only enabling the Variable Block Size feature, report the percentage of blocks that were
    chosen by mode decision to be split into four sub-blocks for every tested QP value. Basically, you
    need to produce a curve whose x-axis is the tested QP value, and the y-axis is the percentage of
    blocks within the sequence that were chosen to be split.
    
    Replace the x-axis (QP value) with bitsteam size and produce another curve.
    """
    pass

def a2_q3_experiment():
    """
    For the test file “synthetic.yuv” provided in Quercus, and using only QP=4 with no other
    experiments, plot a per-frame distortion and encoded bitstream size graphs for each varying
    setting of nRefFrames, from 1 to 4. Explain your results.
    """
    pass

def a2_q4_experiment():
    """
    Visualizations: include a simple tool that helps visualizing a P-frame (or an I-frame) of your
    choosing overlaying the variable block size decisions on top. Suggestion (left):
    """
    pass

def run_tests():
    e4_test()
