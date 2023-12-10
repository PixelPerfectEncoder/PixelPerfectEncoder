import time
import numpy as np
import decimal
from multiprocessing import Manager, shared_memory, Pool
import matplotlib.pyplot as plt
from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.StreamProducer import StreamProducer

videos = {
    "garden": ("garden.yuv", 240, 352),
    "foreman": ("foreman_cif-1.yuv", 288, 352),
    "synthetic": ("synthetic.yuv", 288, 352),
    "CIF": ("CIF.yuv", 288, 352),
    "QCIF": ("QCIF.yuv", 144, 176),
}

def play_CIF(config):
    producer = StreamProducer(videos["CIF"], config)
    return producer.get_stream(play_video=True)
    
def cpu_intensive_task(total_jobs):
    total = 0
    for _ in range(total_jobs):
        decimal.getcontext().prec = 20000
        sqrt_two = decimal.Decimal(2).sqrt()
        for v in str(sqrt_two):
            if v != '.':
                total += int(v)
    return total

def true_cpu_parallelism_verfication():
    def two_threads(jobs):
        start = time.time()
        num_processes = 2
        pool = Pool(processes=num_processes)
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

def worker(args):
    height, width = 288, 352
    ref_frame_mem_name, current_frame_mem_name, queue, is_first_thread = args
    ref_frame_mem = shared_memory.SharedMemory(name=ref_frame_mem_name)
    current_frame_mem = shared_memory.SharedMemory(name=current_frame_mem_name)
    ref_frame = np.ndarray((height, width), dtype=np.uint8, buffer=ref_frame_mem.buf)
    current_frame = np.ndarray((height, width), dtype=np.uint8, buffer=current_frame_mem.buf)
    for row in range(height):
        for col in range(width):
            if col == 0:
                if is_first_thread:
                    if row != 0:
                        print(f"queue.put({row - 1})")
                        queue.put(row - 1)
                else:         
                    if row + 1 < height:         
                        while True:
                            if not queue.empty():
                                finished_row = queue.get()
                                print(f"finished_row={finished_row}")
                                if finished_row == row + 1:
                                    break
                            else:    
                                time.sleep(0.001)
            if is_first_thread:
                current_frame[row, col] = ref_frame[row, col] + row + col
            else:
                current_frame[row, col] = ref_frame[row, col] * 2
    if is_first_thread:
        queue.put(height - 1)

def mode3_concept_verification():
    height, width = 288, 352            
    first_frame_mem = shared_memory.SharedMemory(create=True, size=height * width)
    data = np.ndarray((height, width), dtype=np.uint8, buffer=first_frame_mem.buf)
    data.fill(1)
    second_frame_mem = shared_memory.SharedMemory(create=True, size=height * width)
    thrid_frame_mem = shared_memory.SharedMemory(create=True, size=height * width)
    manager = Manager()
    queue = manager.Queue()
    first_thread = (
        first_frame_mem.name,
        second_frame_mem.name,
        queue,
        True
    )
    second_thread = (
        second_frame_mem.name,
        thrid_frame_mem.name,
        queue,
        False
    )
    pool = pool.Pool(processes=2)
    pool.map(worker, [first_thread, second_thread])
    first_frame = np.ndarray((height, width), dtype=np.uint8, buffer=first_frame_mem.buf)
    second_frame = np.ndarray((height, width), dtype=np.uint8, buffer=second_frame_mem.buf)
    third_frame = np.ndarray((height, width), dtype=np.uint8, buffer=thrid_frame_mem.buf)
    print(first_frame)
    print("=====================================")
    print(second_frame)
    print("=====================================")
    print(third_frame)

def plot_a_RD_to_bitrate_curve(video_name, config: CodecConfig, label: str, show_time=False, total_frames=5):
    qps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R_D = []
    total_time = 0
    for qp in qps:
        config.qp = qp
        producer = StreamProducer(videos[video_name], config)
        start_time = time.time()
        video_data = producer.get_stream(play_video=True, total_frames=total_frames)
        R_D.append((video_data.bitrate, video_data.psnr))
        print(f"label={label}, qp={qp} done")
    total_time = time.time() - start_time
    x = [R_D[i][0] for i in range(len(qps))]
    y = [R_D[i][1] for i in range(len(qps))]
    if show_time:
        average_time = total_time / len(qps)
        label += f" (average time: {average_time:.2f}s)"
    plt.plot(x, y, label=label, linewidth=0.5)

def e3_compare_mode0_mode1():
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
    )
    config.ParallelMode = 0
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 0", True)
    config.ParallelMode = 1
    config.num_processes = 2
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 1 with 2 process", True)
    plt.legend()
    plt.show()
    plt.savefig("e3_compare_mode0_mode1.png")


def e3_compare_mode0_mode1():
    config = CodecConfig(
        block_size=16,
        FastME=True,
        FastME_LIMIT=16,
        FMEEnable=True,
        VBSEnable=False,
        RD_lambda=0.3,
        nRefFrames=1,
        i_Period=10,
        qp=5,
    )
    config.ParallelMode = 0
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 0", True)
    config.ParallelMode = 1
    config.num_processes = 2
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 1 with 2 process", True)
    config.ParallelMode = 1
    config.num_processes = 4
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 1 with 4 process", True)
    config.ParallelMode = 2
    config.num_processes = 2
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 2 with 2 process", True)
    config.ParallelMode = 2
    config.num_processes = 4
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 2 with 4 process", True)
    config.ParallelMode = 3
    config.num_processes = 2
    plot_a_RD_to_bitrate_curve("CIF", config, "Mode 3 with 2 process", True)
    plt.legend()
    plt.show()
    plt.savefig("e3_compare_all_modes.png")
    
def run_e3():
    config = CodecConfig(
        block_size=16,
        FastME=True,
        FastME_LIMIT=16,
        FMEEnable=True,
        VBSEnable=True,
        RD_lambda=0.3,
        nRefFrames=1,
        i_Period=3,
        qp=5,
        ParallelMode=1,
    )
    def run_and_show_time(config):
        start = time.time()
        play_CIF(config)
        print(f"Time taken: {time.time() - start:.2f}s, ParallelMode={config.ParallelMode}, num_processes={config.num_processes}")    
    
    config.ParallelMode = 3
    config.num_processes = 2
    run_and_show_time(config)
    
    config.ParallelMode = 0
    config.num_processes = 1
    run_and_show_time(config)
    
    config.ParallelMode = 1
    config.num_processes = 2
    run_and_show_time(config)
    
    config.ParallelMode = 2
    config.num_processes = 2
    run_and_show_time(config)
    
    config.ParallelMode = 1
    config.num_processes = 4
    run_and_show_time(config)
    
    config.ParallelMode = 2
    config.num_processes = 4
    run_and_show_time(config)
    
    config.ParallelMode = 1
    config.num_processes = 8
    run_and_show_time(config)
    
    config.ParallelMode = 2
    config.num_processes = 8
    run_and_show_time(config)


    