import numpy as np
from PixelPerfect.Yuv import YuvFrame, ReferenceFrame
from PixelPerfect.ResidualProcessor import ResidualProcessor
from PixelPerfect.CodecConfig import CodecConfig
from bitstring import BitArray, BitStream
from math import log2, floor
from typing import Deque
from PixelPerfect.FileIO import get_media_file_path, dump_json, read_frames, read_json
import cv2
import numpy as np
videos = {
    "garden": ("garden.yuv", 240, 352),
    "foreman": ("foreman_cif-1.yuv", 288, 352),
    "synthetic": ("synthetic.yuv", 288, 352),
    "CIF": ("CIF.yuv", 288, 352),
    "QCIF": ("QCIF.yuv", 144, 176),
}
def detect_faces(y_values):
    # Convert the 2D array to uint8 type
    filename, height, width = videos["CIF"]
    y_values = np.array(y_values, dtype=np.uint8)

    # Load the pre-trained Haarcascades face classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(y_values, scaleFactor=1.3, minNeighbors=5)

    return faces
config = CodecConfig(
        block_size=16,
        nRefFrames=1,
        VBSEnable=False,
        FMEEnable=True,
        FastME=1,
        FastME_LIMIT=16,
        RCflag=0,
        targetBR=2 * 1024,
        fps=30,
        total_frames=21,
        filename='CIF'
    )
# Example usage:
# Load your grayscale 2D array
# Replace 'your_2d_array' with your actual 2D array representing luminance values
filename, height, width = videos["foreman"]
# Detect faces
for seq, frame in enumerate(read_frames(get_media_file_path(filename), height, width, config)):
    faces = detect_faces(frame.data)

# Create an RGB image for visualization
    rgb_image = cv2.cvtColor(frame.data, cv2.COLOR_GRAY2RGB)

# Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Detected Faces', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()