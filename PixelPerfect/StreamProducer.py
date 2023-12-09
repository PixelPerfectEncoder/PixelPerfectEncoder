from PixelPerfect.CodecConfig import CodecConfig
from PixelPerfect.VideoEncoder import VideoEncoder 
from PixelPerfect.VideoDecoder import VideoDecoder
from PixelPerfect.Yuv import ReferenceFrame
from PixelPerfect.FileIO import get_media_file_path, read_frames
from typing import List
from multiprocessing import shared_memory, Manager, Pool
from PixelPerfect.CompressedVideoData import CompressedVideoData, StreamMetrics
class CoderStateManager:
    def __init__(self, config: CodecConfig):
        self.config = config
        self.reconstructed_frames: List[ReferenceFrame] = []
        
    def frame_reconstructed(self, frame: ReferenceFrame, frame_seq: int):
        while frame_seq >= len(self.reconstructed_frames):
            self.reconstructed_frames.append(None)
        self.reconstructed_frames[frame_seq] = frame
    
    def is_p_frame(self, frame_seq: int):
        if self.config.i_Period == -1:
            return True
        if self.config.i_Period == 0:
            return False
        if frame_seq % self.config.i_Period == 0:
            return False
        else:
            return True
    
    def is_i_frame(self, frame_seq: int):
        return not self.is_p_frame(frame_seq)
    
    def get_previous_frames(self, frame_seq):
        return self.reconstructed_frames[max(0, frame_seq - self.config.nRefFrames): frame_seq]



class StreamProducer:
    def __init__(self, video, config: CodecConfig):
        self.video = video
        _, height, width = video
        self.config = config
        self.pool = Pool(processes=self.config.num_processes)
        self.shared_mem = shared_memory.SharedMemory(create=True, size=height * width)
    
    
    def get_stream(self, play_video: bool = False, total_frames: int = -1):
        data_stream = []
        filename, height, width = self.video
        encoder = VideoEncoder(height, width, self.config)
        decoder = VideoDecoder(height, width, self.config)
        state = CoderStateManager(self.config)
        psnr_sum = 0
        if self.config.ParallelMode == 0 or self.config.ParallelMode == 1 or self.config.ParallelMode == 2:
            for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, self.config)):
                if total_frames != -1 and seq >= total_frames:
                    break
                compressed_data = encoder.process(frame, state.is_i_frame(seq), state.get_previous_frames(seq), self.shared_mem.name, self.pool)
                data_stream.append(compressed_data)
                decoded_frame = decoder.process(compressed_data, state.is_i_frame(seq), state.get_previous_frames(seq))
                if play_video:
                    decoded_frame.display()
                psnr_sum += frame.PSNR(decoded_frame)
                state.frame_reconstructed(decoded_frame, seq)
        elif self.config.ParallelMode == 3:
            original_frames = [frame for frame in read_frames(get_media_file_path(filename), height, width, self.config)]
            original_frames = original_frames[:total_frames]  
            seq = 0
            another_shared_mem = shared_memory.SharedMemory(create=True, size=height * width)
            while seq < len(original_frames):
                manager = Manager()
                queue = manager.Queue()
                if state.is_i_frame(seq) or seq + 1 == len(original_frames):
                    compressed_data = encoder.mode_3_process(
                        (
                            original_frames[seq], 
                            state.is_i_frame(seq), 
                            state.get_previous_frames(seq), 
                            self.shared_mem.name, 
                            None,
                            queue,
                            True,
                            None
                        )
                    )
                    data_stream.append(compressed_data)
                    decoded_frame = decoder.process(compressed_data, state.is_i_frame(seq), state.get_previous_frames(seq))
                    if play_video:
                        decoded_frame.display()
                    psnr_sum += original_frames[seq].PSNR(decoded_frame)
                    state.frame_reconstructed(decoded_frame, seq)
                    seq += 1
                    continue
    
                first_thread = (
                    original_frames[seq], 
                    state.is_i_frame(seq), 
                    state.get_previous_frames(seq), 
                    self.shared_mem.name, 
                    None, 
                    queue,
                    True,
                    None
                )
                second_thread = (
                    original_frames[seq + 1],
                    state.is_i_frame(seq + 1),
                    [],
                    another_shared_mem.name,
                    None,
                    queue,
                    False,
                    self.shared_mem.name # the name of the reference frame shared memory
                )    
                results = self.pool.map(encoder.mode_3_process, [first_thread, second_thread])
                first_data, second_data = results[0], results[1]
                decoded_first_frame = decoder.process(first_data, state.is_i_frame(seq), state.get_previous_frames(seq))
                state.frame_reconstructed(decoded_first_frame, seq)
                decoded_second_frame = decoder.process(second_data, state.is_i_frame(seq + 1), state.get_previous_frames(seq + 1))
                state.frame_reconstructed(decoded_second_frame, seq + 1)
                data_stream.append(first_data)
                data_stream.append(second_data)
                if play_video:
                    decoded_first_frame.display()
                    decoded_second_frame.display()
                psnr_sum += original_frames[seq].PSNR(decoded_first_frame)
                psnr_sum += original_frames[seq + 1].PSNR(decoded_second_frame)
                seq += 2
        else:
            raise Exception("Unknown ParallelMode")
        return CompressedVideoData(
                config=self.config, 
                stream=data_stream, 
                metrics=StreamMetrics(
                    psnr=psnr_sum / len(data_stream), 
                    bitrate=encoder.bitrate
                )
            )