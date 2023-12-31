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

def plot_a_RD_to_bitrate_curve_use_bitrate_controller(video_name, bitrates, config: CodecConfig, label: str, show_time=False, display=True):
    last_seq = 20
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
                # print(f"{encoder.bitrate} {frame.PSNR(decoded_frame)}")
                R_D.append((encoder.bitrate, (psnr_sum) / (last_seq + 1)))
                break
            print(encoder.frame_bitrate/(br*1024*1024//30))
        print(encoder.bitrate/(br*1024*1024*21//30))
    # x = [R_D[i][0] for i in range(len(bitrates))]
    # y = [R_D[i][1] for i in range(len(bitrates))]
    # if show_time:
    #     average_time = total_time / len(bitrates)
    #     label += f" (average time: {average_time:.2f}s)"
    # plt.plot(x, y, label=label, linewidth=0.5)

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

def get_iframe_threshold():
    config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
    )
    filename, height, width = videos["CIF"]
    frame_type_data = dict()
    for i_p in [-1]:
        qp_data = dict()
        config.i_Period = i_p
        threshold = dict()
        for qp in range(12):
            config.qp = qp
            encoder = VideoEncoder(height, width, config)
            total_frames = 0
            p = []
            i = []
            for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                compressed_data = encoder.process(frame)
                print(encoder.frame_bitrate)
                if seq % 7 == 0:
                    i.append(encoder.frame_bitrate)
                else:
                    p.append(encoder.frame_bitrate)
                prev_bit = encoder.bitrate
                total_frames += 1
            threshold[qp] = (min(i) + max(p))/2
        print(threshold)


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
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
    )
    video_data = dict()
    for video_name in ["CIF", "QCIF"]:
        filename, height, width = videos[video_name]
        frame_type_data = dict()
        for i_p in [1, -1]:
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
    
    dump_json(video_data, "e1_table_vbs_false.json")
    
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
def a3_e1_test():
    config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
        RCflag=0,
        RCTable = read_json("e1_table_vbs_false.json")["CIF"],
        targetBR=2.4 * 1024,
        fps=30,
        total_frames = 21,
        filename = 'CIF'
    )
    filename, height, width = videos["CIF"]
    frame_type_data = dict()
    plt.figure()
    for i_p in [21]:
        config.i_Period = i_p
        threshold = dict()
        bitcount_list = []
        PSNR_list = []
        for qp in [4]:
            config.qp = qp
            encoder = VideoEncoder(height, width, config)
            decoder = VideoDecoder(height, width, config)
            total_frames = 0
            for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                compressed_data = encoder.process(frame)
                print(encoder.frame_seq)
                decoded_frame = decoder.process(compressed_data)
                print(encoder.frame_bitrate)
                bitcount_list.append(encoder.frame_bitrate)
                PSNR_list.append(frame.get_psnr(decoded_frame))
                print(encoder.frame_bitrate / (2.4 * 1024 * 1024 // 30))
                total_frames += 1
        print(sum(bitcount_list)/(2.4 * 1024 * 1024 *21 // 30))
        x = [i for i in range(1,22)]
        y = [bitcount_list[i] for i in range(len(bitcount_list))]
        plt.xlabel("frame_sequence")
        plt.ylabel("bitcounts")
        plt.plot(x, y, label = "i_period="+str(i_p), linewidth=0.5)
        plt.legend(loc='best')

    plt.savefig("PSNR_perframe_QCIF.png")

def a3_e2_test():
    config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
        RCflag=0,
        RCTable = read_json("e1_table_vbs_false.json")["CIF"],
        targetBR=2.4 * 1024,
        fps=30,
        total_frames = 21,
        filename = 'CIF'

    )
    filename, height, width = videos["CIF"]
    frame_type_data = dict()
    plt.figure()
    config.i_Period = 21
    time_list = []
    for RC in [0,1,2,3]:
        if RC==0:
            config.RCflag = 0
            PSNR_ = []
            bitcount_list = []
            for qp in [3,6,9]:
                time_now = time.time()
                config.qp = qp
                encoder = VideoEncoder(height, width, config)
                decoder = VideoDecoder(height, width, config)
                PSNR_list = []
                for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                    compressed_data = encoder.process(frame)
                    decoded_frame = decoder.process(compressed_data)
                    PSNR_list.append(frame.get_psnr(decoded_frame))
                time_list.append(time.time()-time_now)
                bitcount_list.append(encoder.bitrate)
                PSNR_.append(sum(PSNR_list)/len(PSNR_list))
            x = [bitcount_list[i] for i in range(len(bitcount_list))]
            y = [PSNR_[i] for i in range(len(PSNR_))]
            plt.xlabel("bitcounts")
            plt.ylabel("PSNR")
            plt.plot(x, y, label='RC=0', linewidth=0.5)
        else:
            config.RCflag = RC
            bitcount_list = []
            PSNR_list_ = []
            config.qp = 4
            for size in [0.96,2.4,7]:
                if RC>1:
                    if size == 0.96:
                        config.qp = 6
                    if size == 2.4:
                        config.qp = 3
                    if size == 7:
                        config.qp = 1
                time_now = time.time()
                config.targetBR = size*1024
                encoder = VideoEncoder(height, width, config)
                PSNR_list = []
                for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                    compressed_data = encoder.process(frame)
                    print(encoder.frame_seq)
                    print(encoder.frame_bitrate)
                    PSNR_list.append(frame.get_psnr(encoder.previous_frames[0]))
                    print(encoder.frame_bitrate / (size * 1024 * 1024 // 30))
                bitcount_list.append(encoder.bitrate)
                PSNR_list_.append(sum(PSNR_list)/len(PSNR_list))
                time_list.append(time.time() - time_now)
            y = [PSNR_list_[i] for i in range(len(PSNR_list_))]
            x = [bitcount_list[i] for i in range(len(bitcount_list))]
            plt.xlabel("bitcounts")
            plt.ylabel("PSNR")
            plt.plot(x, y, label = "RC="+str(RC), linewidth=0.5)

    print(time_list)
    plt.legend(loc='best')
    plt.savefig("R-D.png")
    plt.show()
def a3_e2_test_part2():
    config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
        RCflag=0,
        RCTable = read_json("e1_table_vbs_false.json")["CIF"],
        targetBR=2 * 1024,
        fps=30,
        total_frames = 21,
        filename = 'CIF'
    )
    filename, height, width = videos["CIF"]
    frame_type_data = dict()
    plt.figure()
    config.i_Period = 21
    for RC in [1,2,3]:
        config.RCflag = RC
        threshold = dict()
        bitcount_list = []
        PSNR_list = []
        for qp in [4]:
            config.qp = qp
            encoder = VideoEncoder(height, width, config)
            decoder = VideoDecoder(height, width, config)
            total_frames = 0
            for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                compressed_data = encoder.process(frame)
                print(encoder.frame_seq)
                # decoded_frame = decoder.process(compressed_data)
                print(encoder.frame_bitrate)
                bitcount_list.append(encoder.frame_bitrate)
                PSNR_list.append(frame.get_psnr(encoder.previous_frames[0]))
                print(encoder.frame_bitrate / (2 * 1024 * 1024 // 30))
                total_frames += 1
        x = [i for i in range(1,22)]
        y = [PSNR_list[i] for i in range(len(PSNR_list))]
        plt.xlabel("frame_sequence")
        plt.ylabel("PSNR")
        plt.plot(x, y, label = "RC="+str(RC), linewidth=0.5)
        plt.legend(loc='best')
    plt.show()
    plt.savefig("PSNR_perframe_QCIF.png")

def facial_test():
    config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
        RCflag=4,
        RCTable=read_json("e1_table_vbs_false.json")["CIF"],
        targetBR=7 * 1024,
        fps=30,
        total_frames=21,
        filename='CIF',
        dQPLimit = 3,
        i_Period=21
        # facial_recognition=True,
    )
    filename, height, width = videos["CIF"]
    frame_type_data = dict()
    plt.figure()
    bitcount_list = []
    PSNR_list = []
    for BR in [7]:
        config.targetBR = BR*1024
        threshold = dict()
        for qp in [6]:
            config.qp = qp
            encoder = VideoEncoder(height, width, config)
            decoder = VideoDecoder(height, width, config)
            total_frames = 0
            PSNR = 0
            bitcount = 0
            for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
                compressed_data = encoder.process(frame)
                decoded_frame = decoder.process(compressed_data)
                decoded_frame.display()
                PSNR += frame.get_psnr(decoded_frame)
                bitcount+=encoder.frame_bitrate
        PSNR_list.append(PSNR/21)
        bitcount_list.append(bitcount/21)
        print(bitcount_list[-1] / (BR * 1024 * 1024 // 30))
    x = [bitcount_list[i] for i in range(len(bitcount_list))]
    y = [PSNR_list[i] for i in range(len(PSNR_list))]
    plt.xlabel("bitcounts")
    plt.ylabel("PSNR")
    plt.plot(x, y,  linewidth=0.5)
    plt.savefig("R_D_ROI.png")
def run_e1():
    # create_e1_table()
    # config = CodecConfig(
    #     block_size=16,
    #     block_search_offset=2,
    #     RCflag = 2,
    #     RCTable = read_json("e1_table.json")["CIF"],
    #     fps = 30,
    #     total_frames = 21,
    #     targetBR=2.4*1024
    # )
    # for i_p in [4]:
    #     config.i_Period = i_p
    #     plot_a_RD_to_bitrate_curve_use_bitrate_controller("CIF", [2.4],config, f"i_Period={i_p}")
    # plt.legend()
    # plt.show()
    
    facial_test()
    # get_iframe_threshold()
            