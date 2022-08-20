# used to handle image observations saved as np arrays
from component import _init_wrapper
from observations.observation import Observation
from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow, show, close
from os import mkdir
from os.path import join, exists
import numpy as np

class Image(Observation):

    # constructor
    @_init_wrapper
    def __init__(self, _data, data_path=None, is_gray=False, timestamp=None):
        super().__init__(_data=_data, data_path=data_path, timestamp=timestamp)

    # displays observation to console
    def display(self):
        if self.is_gray:
            imshow(self._data, cmap='gray')
        else:
            imshow(self._data)
        show()
        close('all')

    # also will write an image to folder
    def write(self, directory_path, file_component=None, as_img=True):
        file_path = super().write(directory_path, file_component)
        if as_img:
            if self.is_gray:
                imsave(file_path + '.png', self._data, cmap='gray')
            else:
                imsave(file_path + '.png', self._data)
        self.data_path = file_path + '.png'