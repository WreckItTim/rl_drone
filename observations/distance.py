# used to handle image observations saved as np arrays
from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow, show, close
from os import mkdir
from os.path import join, exists
import numpy as np
import pickle
import numpy as np
import utils
from observations.observation import Observation

class Distance(Observation):

	# constructor
	def __init__(self, _data, data_path=None):
		super().__init__(_data=_data, data_path=data_path)

	# displays observation to console
	def display(self):
		utils.speak(self._data)

	# also will write an image to folder
	def write(self, directory_path=None, file_name=None):
		file_path = super().write(directory_path=directory_path, file_name=file_name)
		self.data_path = file_path + '.num'
		pickle.dump(self._data, open(self.data_path, 'wb'))

	# turns observation in numpy array to be stacked with other arrays
	def to_stack(self, output_height, output_width):
		stack_shape = (output_height, output_width, 1)
		return np.full(stack_shape, self._data)