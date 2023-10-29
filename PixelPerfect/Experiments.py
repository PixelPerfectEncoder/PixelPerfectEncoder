from PixelPerfect.Decoder import Decoder
from PixelPerfect.Encoder import Encoder, CodecConfig
from PixelPerfect.Yuv import YuvInfo
from PixelPerfect.FileIO import get_media_file_path, dump, load, clean_data, read_frames
import matplotlib.pyplot as plt


filename = "foreman_cif-1.yuv"
video_info = YuvInfo(height=288, width=352)

def varying_block_size_test(block_search_offset, approximated_residual_n):
    blocksize2psnr = dict()
    blocksize2mae = dict()
    for block_size in [2, 4, 8, 16]:
        config = CodecConfig(
            block_size=block_size,
            block_search_offset=block_search_offset,
            approximated_residual_n=approximated_residual_n,
            i_Period=-1,
            do_approximated_residual=True,
            do_dct=False,
            do_quantization=False,
            do_entropy=False,
        )
        encoder = Encoder(video_info, config)
        decoder = Decoder(video_info, config)
        psnr = []
        mae = []
        for frame in read_frames(get_media_file_path(filename), video_info, config):
            compressed_data = encoder.process(frame)
            decoded_frame = decoder.process(compressed_data)
            psnr.append(decoded_frame.get_psnr(frame))
            mae.append(encoder.average_mae)
        blocksize2psnr[block_size] = psnr
        blocksize2mae[block_size] = mae
    plt.figure()
    for block_size, psnr in blocksize2psnr.items():
        plt.plot(psnr, label=f'block_size={block_size}')
    plt.legend()
    plt.title('PSNR vs Frame Index')
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR')
    plt.show()


def e3_1_report():
    """
    A per-frame PSNR graph measured as PSNR between original and reconstructed frames as well as
    a per-frame average MAE graph calculated during the MV selection process. Show a set of graphs
    for varying 𝑖 with fixed 𝑟=4 and 𝑛=3, another for varying 𝑟 with fixed 𝑖=8 and 𝑛=3, and another
    for varying 𝑛 with fixed 𝑖=8 and 𝑟=4. You can use any sequences you want, but the first 10 frames
    of Foreman CIF (352x288) are a requirement, plus at least one other sequence of different
    dimensions (number of frames at your discretion).
    """
    varying_block_size_test(4, 3)
    