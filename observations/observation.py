# abstract class used to handle observations - sensor responses
from component import Component, _init_wrapper
from utils import get_timestamp, write_json, fix_directory
from os import mkdir
from os.path import exists
import numpy as np

class Observation(Component):
    # constructor
    def __init__(self, _data, data_path=None, timestamp=None):
        if timestamp is None:
            self.timestamp = get_timestamp()
        if data_path is not None:
            _data = self.read_data(data_path)
        self._data = np.array(_data)
        self.transformations = []
        self._name = 'Observation_' + self.timestamp

    # writes observation metadata to given dir path
    def write(self, directory_path, file_name=None):
        directory_path = fix_directory(directory_path)
        if file_name is None:
            file_name = f'{type(self).__name__}__{self.timestamp}'
        file_path = directory_path + '/' + file_name
        if not exists(directory_path):
            mkdir(directory_path)
        serialized = self._to_json()
        write_json(serialized, file_path)
        return file_path

    def activate(self):
        self.display()

    # displays observation to console
    def display(self):
        raise NotImplementedError

    # converts observation into an nd numpy array
    def to_numpy(self):
        return self._data

    # sets numpy array data
    def set_data(self, data):
        self._data = data

    # saves transformation info (after transformation is done)
    def save_transformation(self, transformer, data):
        self.transformations.append(transformer._to_json())
        self.set_data(data)