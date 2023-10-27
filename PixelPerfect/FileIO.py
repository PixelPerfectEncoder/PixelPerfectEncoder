import uuid
import pickle
import os
import numpy as np
from PixelPerfect.Yuv import YuvInfo, YuvFrame


def read_frames(path, info: YuvInfo):
    height = info.height
    width = info.width
    yuv_frame_size = width * height + (width // 2) * (height // 2) * 2
    y_frame_size = width * height

    with open(path, "rb") as file:
        while True:
            yuv_frame_data = file.read(yuv_frame_size)
            if len(yuv_frame_data) < yuv_frame_size:
                break
            yield YuvFrame(
                np.frombuffer(yuv_frame_data[:y_frame_size], dtype=np.uint8).reshape(
                    (info.height, info.width)
                )
            )


def get_media_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join("media", filename))


def get_data_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join("data", filename))


def dataId2filename(id):
    return get_data_file_path(f"{id}.txt")


def dump(data):
    id = uuid.uuid4().hex
    with open(dataId2filename(id), "wb") as file:
        pickle.dump(data, file)
    return id


def load(id):
    with open(dataId2filename(id), "rb") as file:
        return pickle.load(file)


def clean_data(file_ids):
    for id in file_ids:
        os.remove(dataId2filename(id))
