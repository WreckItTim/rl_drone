# used to handle image observations saved as np arrays
from observations.observation import Observation
from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow, show, close
from os import mkdir
from os.path import join, exists
import numpy as np

class Image(Observation):

    # constructor
    def __init__(self, _data, data_path=None, is_gray=False):
        super().__init__(_data=_data, data_path=data_path)
        self.is_gray = is_gray

    # displays observation to console
    def display(self):
        if self.is_gray:
            imshow(self._data, cmap='gray')
        else:
            imshow(self._data)
        show()

    # also will write an image to folder
    def write(self, directory_path=None, file_name=None, as_img=True):
        file_path = super().write(directory_path=directory_path, file_name=file_name)
        self.data_path = file_path + '.png'
        if as_img:
            if self.is_gray:
                imsave(self.data_path, self._data.reshape([self._data.shape[1], self._data.shape[2]]), cmap='gray', vmin=0, vmax=255)
            else:
                imsave(self.data_path, self._data, vmin=0, vmax=255)