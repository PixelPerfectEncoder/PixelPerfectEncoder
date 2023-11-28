from PixelPerfect.Decoder import VideoDecoder
from PixelPerfect.Encoder import VideoEncoder
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.FileIO import get_media_file_path, dump_json, read_frames, read_json
import matplotlib.pyplot as plt
import time

videos = {
    "garden": ("garden.yuv", 240, 352),
    "foreman": ("foreman_cif-1.yuv", 288, 352),
    "synthetic": ("synthetic.yuv", 288, 352),
    "CIF": ("CIF.yuv", 288, 352),
    "QCIF": ("QCIF.yuv", 144, 176),
}

def plot_a_RD_to_bitrate_curve(video_name, config: CodecConfig, label: str, show_time=False, display=True):
    qps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    last_seq = 10
    R_D = []
    total_time = 0
    filename, height, width = videos[video_name]
    for qp in qps:
        config.qp = qp
        encoder = VideoEncoder(height, width, config)
        decoder = VideoDecoder(height, width, config)
        psnr_sum = 0
        for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
            t1 = time.time()
            compressed_data = encoder.process(frame)
            total_time += time.time() - t1
            decoded_frame = decoder.process(compressed_data)
            decoded_frame.display()
            psnr_sum += frame.PSNR(decoded_frame)
            if seq == last_seq:
                print(f"{encoder.bitrate} {frame.PSNR(decoded_frame)}")
                R_D.append((encoder.bitrate, (psnr_sum) / (last_seq + 1)))
                break
    x = [R_D[i][0] for i in range(len(qps))]
    y = [R_D[i][1] for i in range(len(qps))]
    if show_time:
        average_time = total_time / len(qps)
        label += f" (average time: {average_time:.2f}s)"
    plt.plot(x, y, label=label, linewidth=0.5)

def plot_a_RD_to_bitrate_curve_use_bitrate_controller(video_name, config: CodecConfig, label: str, show_time=False, display=True):
    bitrates = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5]
    last_seq = 10
    R_D = []
    total_time = 0
    filename, height, width = videos[video_name]
    for br in bitrates:
        config.targetBR = br * 1024
        encoder = VideoEncoder(height, width, config)
        decoder = VideoDecoder(height, width, config)
        psnr_sum = 0
        for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
            t1 = time.time()
            compressed_data = encoder.process(frame)
            total_time += time.time() - t1
            decoded_frame = decoder.process(compressed_data)
            decoded_frame.display()
            psnr_sum += frame.PSNR(decoded_frame)
            if seq == last_seq:
                print(f"{encoder.bitrate} {frame.PSNR(decoded_frame)}")
                R_D.append((encoder.bitrate, (psnr_sum) / (last_seq + 1)))
                break
    x = [R_D[i][0] for i in range(len(bitrates))]
    y = [R_D[i][1] for i in range(len(bitrates))]
    if show_time:
        average_time = total_time / len(bitrates)
        label += f" (average time: {average_time:.2f}s)"
    plt.plot(x, y, label=label, linewidth=0.5)

def simple_test():
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
    )
    for i_p in [4]:
        config.i_Period = i_p
        plot_a_RD_to_bitrate_curve("CIF", config, f"i_Period={i_p}")
    plt.legend()
    plt.show()

def create_e1_table():
    """
    Configure your encoder to operate with (nRefFrames=1, VBSEnable=1, FMEEnable=1, and FastME=1),
    using block size = 16x16, bounding the MVs to be within a range of +/-16 pixels around the collocated
    block, and other optimization modes at your discretion. Obtain the corresponding tables for part (a),
    based on statistics collected from the provided artificially-generated CIF and QCIF sequences (a 21-frame
    artificial sequence composed of the first 7 frames of Foreman CIF/QCIF, followed by the first 7 frames of
    Akiyo CIF/QCIF, followed by the second 7 frames of Foreman CIF/QCIF). In order to fill the tables, you will
    be averaging bitcount throughout the 21 frames. The table(s) that you use in part (c) should include all QP
    values from 0 to 11. Ideally, the average bitcount that corresponds to every QP value in the table should
    be measured statistically. However, practically, if you believe that the stats-collection phase would be too
    time-consuming, you can interpolate every other QP bit-count by averaging the two direct neighboring
    measured bit-counts (e.g., bitcount of QP1 = [bitcount of QP0+ bitcount of QP2]/2). You will be producing
    4 tables in total: CIF and QCIF, I-Frame and P-Frame for each. To get I-Frame statistics, you can encode
    with an I_Period of 1 (only I-frames) and take the average. To get P-Frame statistics, use an I_Period of 21
    and average the bitcount of the corresponding 20 P-Frames
    """
    config = CodecConfig(
        block_size=16,
        do_dct=True,
        do_quantization=True,
        nRefFrames=1,
        VBSEnable=True,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
    )
    video_data = dict()
    for video_name in ["CIF", "QCIF"]:
        filename, height, width = videos[video_name]
        frame_type_data = dict()
        for i_p in [1, 21]:
            qp_data = dict()
            config.i_Period = i_p
            for qp in range(12):
                config.qp = qp
                encoder = VideoEncoder(height, width, config)
                total_frames = 0
                for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                    compressed_data = encoder.process(frame)
                    total_frames += 1
                bit_count = encoder.bitrate
                bit_count /= total_frames
                bit_count /= (height / 16)
                print(f"{video_name} i_p={i_p} qp={qp} bit_count={bit_count}")
                qp_data[qp] = bit_count
            if i_p == 1:
                frame_type_data["I"] = qp_data
            else:
                frame_type_data["P"] = qp_data
        video_data[video_name] = frame_type_data
    
    dump_json(video_data, "e1_table.json")
        
 
def run_e1():
    print(read_json("e1_table.json"))
    config = CodecConfig(
        block_size=16,
        block_search_offset=2,
        RCflag = 1,
        RCTable = read_json("e1_table.json")["CIF"],
        fps = 30,
        total_frames = 21,
        targetBR=2.4*1024
    )
    for i_p in [4]:
        config.i_Period = i_p
        plot_a_RD_to_bitrate_curve_use_bitrate_controller("CIF", config, f"i_Period={i_p}")
    plt.legend()
    plt.show()
            