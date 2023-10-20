import cv2
import numpy as np
import os

def play_yuv(filename, width, height):
    with open(filename, 'rb') as file:
        frame_size = width * height + (width // 2) * (height // 2) * 2
        while True:
            yuv_frame = file.read(frame_size)
            if len(yuv_frame) < frame_size:
                break            
            yuv_data = np.frombuffer(yuv_frame, dtype=np.uint8)
            
            y_frame = yuv_data[:width*height].reshape((height, width))
            uv_frame = yuv_data[width*height:].reshape((2, height//2, width//2))
            
            u_channel = cv2.resize(uv_frame[0,:,:], (width, height), interpolation=cv2.INTER_LINEAR)
            v_channel = cv2.resize(uv_frame[1,:,:], (width, height), interpolation=cv2.INTER_LINEAR)

            
                        
            # Stack the Y and UV frames to create a complete frame
            yuv_image = cv2.merge([y_frame, u_channel, v_channel])
            
            # Convert YUV to BGR
            bgr_frame = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            
            # Display the frame
            cv2.imshow('Frame', bgr_frame)
            
            # Wait for the user to press 'q' or the window to be closed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

from .Decoder import Decoder
from .Encoder import Encoder, EncoderConfig
from .Yuv import YuvVideo, YuvMeta

if __name__ == '__main__':
    filename = 'foreman_cif-1.yuv'
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_dir, os.path.join('media', filename))
    video_info = YuvMeta(height=288, width=352)
    video = YuvVideo(file_path, video_info)
    config = EncoderConfig(block_size=16, block_search_offset=2)
    encoder = Encoder(video, config)
    decoder = Decoder(video_info, config)
    for compressed_data in encoder.process():
        decoded_frame = decoder.process(compressed_data)
        decoded_frame.play()
        