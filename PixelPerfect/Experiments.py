from PixelPerfect.Decoder import Decoder
from PixelPerfect.Encoder import Encoder, CodecConfig
from PixelPerfect.FileIO import get_media_file_path, dump, load, clean_data, read_frames
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

videos = {
    "garden": ("garden.yuv", 240, 352),
    "foreman": ("foreman_cif-1.yuv", 288, 352),
}

filename, height, width = videos["foreman"]


def e3_1_report_run_once(i, r, n, total_frames=10):
    print(f"block_size={i}")
    config = CodecConfig(
        block_size=i,
        block_search_offset=r,
        approximated_residual_n=n,
        i_Period=-1,
        do_approximated_residual=True,
        do_dct=False,
        do_quantization=False,
        do_entropy=False,
    )
    encoder = Encoder(height, width, config)
    decoder = Decoder(height, width, config)
    psnr = []
    mae = []
    for seq, frame in enumerate(
        read_frames(get_media_file_path(filename), height, width, config)
    ):
        if seq == total_frames:
            break
        print(f"frame={seq}")
        compressed_data = encoder.process(frame)
        decoded_frame = decoder.process(compressed_data)
        psnr.append(decoded_frame.get_psnr(frame))
        mae.append(encoder.average_mae)
    return psnr, mae


def varying_i(r, n, total_frames=10):
    i2psnr = dict()
    i2mae = dict()
    for i in [4, 8, 16]:
        i2psnr[i], i2mae[i] = e3_1_report_run_once(i, r, n, total_frames)
    plt.figure()
    for i, psnr in i2psnr.items():
        plt.plot(psnr, label=f"block_size={i}")
    plt.legend()
    plt.title("PSNR vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR")
    plt.savefig(
        f"{filename} PSNR varying_i, r={r}, n={n}, total_frames={total_frames}.png",
        dpi=300,
    )

    plt.figure()
    for i, mae in i2mae.items():
        plt.plot(mae, label=f"block_size={i}")
    plt.legend()
    plt.title("MAE vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("MAE")
    plt.savefig(
        f"{filename} MAE varying_i, r={r}, n={n}, total_frames={total_frames}.png",
        dpi=300,
    )


def varing_r(i, n, total_frames=10):
    r2psnr = dict()
    r2mae = dict()
    for r in [2, 4, 6]:
        r2psnr[r], r2mae[r] = e3_1_report_run_once(i, r, n, total_frames)
    plt.figure()
    for r, psnr in r2psnr.items():
        plt.plot(psnr, label=f"block_search_offset={r}")
    plt.legend()
    plt.title("PSNR vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR")
    plt.savefig(
        f"{filename} PSNR varying_r, i={i}, n={n}, total_frames={total_frames}.png",
        dpi=300,
    )

    plt.figure()
    for r, mae in r2mae.items():
        plt.plot(mae, label=f"block_search_offset={r}")
    plt.legend()
    plt.title("MAE vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("MAE")
    plt.savefig(
        f"{filename} MAE varying_r, i={i}, n={n}, total_frames={total_frames}.png",
        dpi=300,
    )


def varying_n(i, r, total_frames=10):
    n2psnr = dict()
    n2mae = dict()
    for n in [1, 2, 3]:
        n2psnr[n], n2mae[n] = e3_1_report_run_once(i, r, n, total_frames)
    plt.figure()
    for n, psnr in n2psnr.items():
        plt.plot(psnr, label=f"approximated_residual_n={n}")
    plt.legend()
    plt.title("PSNR vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR")
    plt.savefig(
        f"{filename} PSNR varying_n, i={i}, r={r}, total_frames={total_frames}.png",
        dpi=300,
    )

    plt.figure()
    for n, mae in n2mae.items():
        plt.plot(mae, label=f"approximated_residual_n={n}")
    plt.legend()
    plt.title("MAE vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("MAE")
    plt.savefig(
        f"{filename} MAE varying_n, i={i}, r={r}, total_frames={total_frames}.png",
        dpi=300,
    )


def e3_1_report():
    """
    A per-frame PSNR graph measured as PSNR between original and reconstructed frames as well as
    a per-frame average MAE graph calculated during the MV selection process. Show a set of graphs
    for varying 𝑖 with fixed 𝑟=4 and 𝑛=3, another for varying 𝑟 with fixed 𝑖=8 and 𝑛=3, and another
    for varying 𝑛 with fixed 𝑖=8 and 𝑟=4. You can use any sequences you want, but the first 10 frames
    of Foreman CIF (352x288) are a requirement, plus at least one other sequence of different
    dimensions (number of frames at your discretion).
    """
    varying_i(4, 3)
    varing_r(8, 3)
    varying_n(8, 4)


def e3_2_report():
    """
    For the 𝑖=8, 𝑟=4 and 𝑛=3 case, include a Y-only reconstructed file of the first 10 frames of Foreman
    CIF (this file should be exactly 1,013,760 bytes in size). Include also a text file with the found
    motion vectors (the format of this file is arbitrary, but it should contain 15,840 𝑥/𝑦 pairs).
    """
    import numpy as np

    def save_as_yuv(data, filename):
        with open(filename, "wb") as f:
            for frame in data:
                f.write(frame.tobytes())
                f.write(np.zeros((144, 176), dtype=np.uint8).tobytes())
                f.write(np.zeros((144, 176), dtype=np.uint8).tobytes())

    config = CodecConfig(
        block_size=8,
        block_search_offset=4,
        approximated_residual_n=3,
        i_Period=-1,
        do_approximated_residual=True,
        do_dct=False,
        do_quantization=False,
        do_entropy=False,
    )
    encoder = Encoder(height, width, config)
    decoder = Decoder(height, width, config)
    motion_vectors = list()
    reconstructed_frames = []
    for seq, frame in enumerate(
        read_frames(get_media_file_path(filename), height, width, config)
    ):
        if seq == 10:
            break
        print(f"frame={seq}")
        compressed_data = encoder.process(frame)
        for block_seq, block_data in enumerate(compressed_data):
            _, row_mv, col_mv = block_data
            row_block_num = encoder.previous_frame.width // 8
            block_row = block_seq // row_block_num * 8
            block_col = block_seq % row_block_num * 8
            motion_vectors.append(
                (
                    f"frame:{seq}, block row:{block_row}, block col:{block_col}",
                    row_mv,
                    col_mv,
                )
            )
        decoded_frame = decoder.process(compressed_data)
        reconstructed_frames.append(decoded_frame.data)

    open(f"motion_vectors {filename} i=8, r=4, n=3.txt", "w").write(
        "\n".join([str(mv) for mv in motion_vectors])
    )
    reconstructed_file_name = f"reconstructed_frames {filename} i=8, r=4, n=3.yuv"
    save_as_yuv(reconstructed_frames, reconstructed_file_name)
    for frame in read_frames(reconstructed_file_name, height, width, config):
        frame.display()


def e3_3_report():
    """
    For at least two 𝑖 cases of your choice, the residual before and after motion compensation, as well
    as the predicted frame before reconstruction, similar to what is shown in the Implementation
    Notes section. You can choose 𝑟, 𝑛, and the frame number as you see fit (but do not choose the
    first frame – the predicted frame must not be all gray)
    """
    # get residual
    r, n = 4, 3
    for i in [4, 16]:
        print(f"block_size={i}")
        config = CodecConfig(
            block_size=i,
            block_search_offset=r,
            approximated_residual_n=n,
            i_Period=-1,
            do_approximated_residual=True,
            do_dct=False,
            do_quantization=False,
            do_entropy=False,
        )
        encoder = Encoder(height, width, config)
        previous_frame = None
        for seq, frame in enumerate(
            read_frames(get_media_file_path(filename), height, width, config)
        ):
            compressed_data = encoder.process(frame)
            if seq == 0:
                previous_frame = encoder.previous_frame
                continue
            previous_data = previous_frame.data.astype(np.int16)
            current_data = frame.data.astype(np.int16)
            residual_before = np.abs(previous_data - current_data).astype(np.uint8)
            residual_after = np.zeros(
                (height, width), dtype=np.uint8
            )
            for seq, block_data in enumerate(compressed_data):
                _, row_mv, col_mv = block_data
                row_block_num = encoder.previous_frame.width // i
                block_row = seq // row_block_num * i
                block_col = seq % row_block_num * i
                residual_after[
                    block_row : block_row + i, block_col : block_col + i
                ] = np.abs(
                    previous_data[
                        block_row + row_mv : block_row + row_mv + i,
                        block_col + col_mv : block_col + col_mv + i,
                    ]
                    - current_data[block_row : block_row + i, block_col : block_col + i]
                )
            Image.fromarray(residual_before).save(
                f"{filename} residual before i={i}.png"
            )
            Image.fromarray(residual_after).save(f"{filename} residual after i={i}.png")
            break


def e3_4_report():
    """
    Given the per-frame PSNR graph measured between original and reconstructed frames and the
    per-frame average MAE graph calculated during the MV selection process in the deliverables,
    which one will show clear variation with 𝑖 and/or 𝑟? and why? Explain the results you are seeing
    """
    pass

def e4_1_report():
    """
    For a fixed set of parameters (𝑖 = 8 and 16; and search range = 2), create R-D plots where the x
    axis is the total size in bits of a test sequence, and the y axis is the quality/distortion measured as
    PSNR. Draw 3 curves: one for an I_Period of 1 (GOP is IIII…), another for an I_Period of 4 (GOP is
    IPPPIPPP…), and finally another for an I_Period of 10 (IPPPPPPPPPIPP…). You get the different
    points of a curve by varying the QP parameter (0.. log2(𝑖)+7 – if this is too much, you can skip
    several and do 0, 3, 6 and 9 for the 8x8 block size, and 1, 4, 7 and 10 for 16x16, but always include
    these four). Use the first 10 frames of Foreman CIF for testing. You can add more sequences of
    your choosing, but present them separately. Since you will be running 12×2 (or more) instances
    of the encoder, it is highly recommended that you write script(s) to run the experiments, collect
    the results, and put them in a tabular format that easily maps into the desired curves. Record (and
    plot) the execution times too.
    """
    
    pass

def e4_2_report():
    """
    For (𝑖=8, QP=3) and (𝑖=16, QP=4) experiments, plot the bit-count (on-y-axis) vs. frame index (on
    x-axis) curves. Show plots for three different values of I_Period (1, 4 and 10).
    """
    pass

