import uuid
import pickle
import os

def get_media_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join('media', filename))

def get_data_file_path(filename):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent_dir, os.path.join('data', filename))

def dataId2filename(id):
    return get_data_file_path(f"{id}.txt")

def dump(data):
    id = uuid.uuid4().hex
    with open(dataId2filename(id), 'wb') as file:
        pickle.dump(data, file)
    return id

def load(id):
    with open(dataId2filename(id), 'rb') as file:
        return pickle.load(file)

def clean_data(file_ids):
    for id in file_ids:
        os.remove(dataId2filename(id))
    