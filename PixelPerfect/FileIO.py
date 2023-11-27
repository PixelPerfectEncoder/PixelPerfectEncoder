import uuid
import pickle
import os
from PixelPerfect.Yuv import ReferenceFrame
from PixelPerfect.CodecConfig import CodecConfig
import numpy as np
import json

def read_frames(source_path, height, width, config: CodecConfig):
    height = height
    width = width
    yuv_frame_size = width * height + (width // 2) * (height // 2) * 2
    y_frame_size = width * height
    
    for yuv_frame_data in read_video(source_path, yuv_frame_size):
        yield ReferenceFrame(
            config,
            data=np.frombuffer(yuv_frame_data[:y_frame_size], dtype=np.uint8).reshape((height, width)),
        )

def read_video(path, size):
    with open(path, "rb") as file:
        while True:
            data = file.read(size)
            if len(data) < size:
                break
            yield data

def get_media_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join("media", filename))


def get_data_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join("data", filename))


def dataId2filename(id):
    return get_data_file_path(f"{id}.txt")

def dump_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)

def dump(data):
    id = uuid.uuid4().hex
    with open(dataId2filename(id), "wb") as file:
        pickle.dump(data, file, indent=4)
    return id


def load(id):
    with open(dataId2filename(id), "rb") as file:
        return pickle.load(file)


def clean_data(file_ids):
    for id in file_ids:
        os.remove(dataId2filename(id))

def get_test_result_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.join(parent_dir, "test_result")
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    return os.path.join(parent_dir, filename)