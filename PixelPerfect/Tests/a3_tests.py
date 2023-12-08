from PixelPerfect.VideoDecoder import VideoDecoder
from PixelPerfect.VideoEncoder import VideoEncoder
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

def plot_a_RD_to_bitrate_curve_use_bitrate_controller(video_name, bitrates, config: CodecConfig, label: str, show_time=False, display=True):
    last_seq = 21
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
    
def paint_e1_table():
    """
    Using the algorithm in part (c) and the same encoder configs you used for stats-collection, encode the
    sequences targeting a bitrate of 2.4 mbps for the CIF sequence, and 960 kbps for the QCIF one, with
    I_Period = 1, 4 and 21. Assume sequences have 30 frames per second. In your report, include:
    
    Per-frame bit cost and PSNR graphs (for each I_Period and sequence)
    """
    config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=True,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
    )
    for i_p in [1, 4, 21]:
        config.i_Period = i_p
        plot_a_RD_to_bitrate_curve_use_bitrate_controller("CIF", config, f"i_Period={i_p}, video=CIF", show_time=True)
        
def vbs_test():
    config = CodecConfig(
        block_size=16,
        i_Period=10,
    )
    plot_a_RD_to_bitrate_curve("foreman", config, label="All features off", show_time=True)
    config.VBSEnable = True
    config.RD_lambda = 0.3
    plot_a_RD_to_bitrate_curve("foreman", config, label="VBSEnable", show_time=True)
    config.VBSEnable = False
    config.FMEEnable = True
    plot_a_RD_to_bitrate_curve("foreman", config, label="FMEEnable", show_time=True)
    config.FMEEnable = False
    config.FastME = True
    config.FastME_LIMIT = 16
    plot_a_RD_to_bitrate_curve("foreman", config, label="FastME", show_time=True)
    config.FastME = False
    config.nRefFrames = 3
    plot_a_RD_to_bitrate_curve("foreman", config, label="nRefFrames = 3", show_time=True)
    config.VBSEnable = True
    config.FMEEnable = True
    config.FastME = True
    config.FastME_LIMIT = 16
    plot_a_RD_to_bitrate_curve("foreman", config, label="All features on", show_time=True)
    plt.legend()
    plt.show()
 
def run_e1():
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


def play_CIF(config):
    filename, height, width = videos["CIF"]
    encoder = VideoEncoder(height, width, config)
    decoder = VideoDecoder(height, width, config)
    for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
        if seq == 5:
            break
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.display()

def cpu_intensive_task(total_jobs):
    import decimal
    total = 0
    for _ in range(total_jobs):
        decimal.getcontext().prec = 20000
        sqrt_two = decimal.Decimal(2).sqrt()
        for v in str(sqrt_two):
            if v != '.':
                total += int(v)
    return total

def true_cpu_parallelism_verfication():
    import multiprocessing
    def two_threads(jobs):
        start = time.time()
        num_processes = 2
        pool = multiprocessing.Pool(processes=num_processes)
        task_parameters = [jobs // num_processes for _ in range(num_processes)]
        pool.map(cpu_intensive_task, task_parameters)
        pool.close()
        pool.join()
        print(f"Time taken: {time.time() - start:.2f}s")


    def one_thread(jobs):
        start = time.time()
        cpu_intensive_task(jobs)
        print(f"Time taken: {time.time() - start:.2f}s")
    
    two_threads(30)
    one_thread(30)

def run_e3():
    config = CodecConfig(
        block_size=16,
        FastME=True,
        FastME_LIMIT=16,
        FMEEnable=True,
        VBSEnable=True,
        RD_lambda=0.3,
        nRefFrames=1,
        i_Period=10,
        qp=5,
        ParallelMode=1,
    )
    
    start = time.time()
    config.ParallelMode = 2
    config.i_Period = 1
    play_CIF(config)
    print(f"Two Threads Time taken: {time.time() - start:.2f}s")
    config.ParallelMode = 0
    start = time.time()
    play_CIF(config)
    print(f"Single Thread Time taken: {time.time() - start:.2f}s")
    config.num_processes = 4
    start = time.time()
    play_CIF(config)
    print(f"Four Threads Time taken: {time.time() - start:.2f}s")
    config.FMEEnable = False
    play_CIF(config)
    config.FMEEnable = True
    config.VBSEnable = False
    play_CIF(config)
        

    