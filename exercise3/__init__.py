import cv2
import numpy as np
import os

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(parent_dir, os.path.join('media', filename))
        self.f = open(file_path, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print (str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return ret, bgr

def extract_Y(file):
    result = []
    while 1:
        ret, frame = file.read()
        if ret:
            frame_Y=[]
            for row in frame:
                row_y = []
                for pixel in row:
                    row_y.append(pixel[1])
                frame_Y.append(row_y)
        else:
            break
        result.append(frame_Y)
    return result

def split_frame(frame, num):
    # do the padding if not divisible
    while len(frame) % num != 0:
        frame.append([128 for i in range(len(frame[0]))])
    if len(frame[0]) % num != 0:
        for row in frame:
            while len(row) % num != 0:
                row.append(128)
    # split the frame
    width = int(len(frame[0]) // num)
    height = int(len(frame) // num)
    splited_frame = []
    for i in range(num):
        for j in range(num):
            sub_frame = []
            for m in range(i*height, (i+1)*height):
                sub_row =[]
                for n in range(j*width, (j+1)*width):
                    sub_row.append(frame[m][n])
                sub_frame.append(sub_row)
            splited_frame.append(sub_frame)
    return splited_frame

def mean_absolute_error(block1, block2):
    error = 0
    for i in range(len(block1)):
        error += abs(block1[i] - block2[i//len(block2[0])][i%len(block2[0])])
    return error

def pixel_full_search_block(ref_frame, cur_block, x, y, rang):
    min_dif = float('inf')
    vector = (0,0)
    for r_x in range(x-rang,x+rang+1):
        if r_x<0 or r_x > (len(ref_frame[0])-len(cur_block[0])):
            continue
        for r_y in range(y-rang, y+rang+1):
            if r_y < 0 or r_y > (len(ref_frame)-len(cur_block)):
                continue
            temp_block = []
            for i in range(r_y, r_y + len(cur_block)):
                for j in range(r_x, r_x + len(cur_block[0])):
                    temp_block.append(ref_frame[i][j])
            error = mean_absolute_error(temp_block, cur_block)
            vec_x = x - r_x
            vec_y = y - r_y
            if error < min_dif:
                vector = (vec_x, vec_y)
                min_dif = error
            elif error == min_dif:
                if abs(vec_x) + abs(vec_y) < abs(vector[0]) + abs(vector[1]):
                    vector = (vec_x, vec_y)
                elif abs(vec_x) + abs(vec_y) == abs(vector[0]) + abs(vector[1]):
                    if abs(vec_y) < abs(vector[1]):
                        vector = (vec_x, vec_y)
                    elif abs(vec_y) == abs(vector[1]):
                        if abs(vec_x) < abs(vector[0]):
                            vector = (vec_x, vec_y)
    index = (x-vector[0], y-vector[1])
    return (vector,index)


def pixel_full_search_frame(ref_frame, cur_frame, div_num, rang):
    block_width = len(cur_frame[0][0])
    block_height = len(cur_frame[0])
    matches = []
    frame_vector = []
    for i in range(len(cur_frame)):
        x = i % div_num
        y = i // div_num
        # if x == 0 or x == div_num-1 or y == 0 or y == div_num-1:
        #     frame_vector.append((0,0))
        #     matches.append((x,y))
        #     continue
        motion_vector = pixel_full_search_block(ref_frame, cur_frame[i], x*block_width, y*block_height, rang)
        frame_vector.append(motion_vector[0])
        matches.append(motion_vector[1])
    return (frame_vector,matches)

def search_whole(ref_frame, video, div_num, search_range):
    result_vectors = []
    result_matches = []
    for frame in video:
        block_frame = split_frame(frame, div_num)
        vectors = pixel_full_search_frame(ref_frame, block_frame, div_num, search_range)
        ref_frame = frame
        result_vectors.append(vectors[0])
        result_matches.append(vectors[1])
    return (result_vectors,result_matches)

def calculate_residual_frame(ref_frame, cur_frame, index):
    residuals_frame = []
    for i in range(len(cur_frame)):
        residual_block = []
        for y in range(len(cur_frame[i])):
            for x in range(len(cur_frame[i][y])):
                try:
                    print(cur_frame[i][y][x])
                    print(ref_frame[index[i][1]+y][index[i][0]+x])
                    residual = int(cur_frame[i][y][x]) - int(ref_frame[index[i][1]+y][index[i][0]+x])
                    residual_block.append(residual)
                except Exception as e:
                    print(index[i][1], y, index[i][0], x)
        residuals_frame.append(residual_block)
    return residuals_frame

def get_residual_frame(ref_frame, video, matches, div_num):
    residuals = []
    for i in range(len(video)):
        block_frame = split_frame(video[i], div_num)
        residuals.append(calculate_residual_frame(ref_frame, block_frame, matches[i]))
        ref_frame = video[i]
    return residuals

# if __name__ == "__main__":
#     filename = "input.yuv"
#     size = (240, 352)
#     div_num = 4
#     h_range = 2
#     ref_frame = [[128 for i in range(352)] for j in range(240)]
#     cap = VideoCaptureYUV(filename, size)
#     y_only = extract_Y(cap)
#     vectors = search_whole(ref_frame, y_only[0:5], div_num, 4)
#     residuals = get_residual_frame(ref_frame, y_only[0:5], vectors[1], div_num)
#     print('finish')
#     while 1:
#         ret, frame = cap.read()
#         if ret:
#             cv2.imshow("frame", frame)
#             cv2.waitKey(30)
#         else:
#             break
        
        


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


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(parent_dir, os.path.join('media', "foreman_cif-1.yuv"))
# Usage
play_yuv(file_path, 352, 288)
