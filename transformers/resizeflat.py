# resizes image as flattened array, converts Image to Vector
from transformers.transformer import Transformer
from observations.vector import Vector
from skimage.transform import resize
import numpy as np
from component import _init_wrapper
import rl_utils as utils
import cv2

class ResizeFlat(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self,
				max_cols=[84], # use only values below given col
				max_rows=[42], # use only values below given row
	):
		super().__init__()
		self._dims = len(max_cols) * len(max_rows)

	# if observation type is valid, applies transformation
	def transform(self, observation):
		img_data = observation.to_numpy()
		# TEMP CODE TODO:DELETE DELETE DELETE
		img_data = np.moveaxis(img_data, 0, 2)
		cv2.imwrite(utils.get_global_parameter('working_directory') + 'tello_imgs/' + observation._name + '_post.png', img_data)
		img_data = np.moveaxis(img_data, 2, 0)
		# TEMP CODE TODO:DELETE DELETE DELETE
		#observation.check(Vector)
		# get max pixel value at each column section
		col_len = len(self.max_cols)

		new_array = np.zeros(self._dims, dtype=float)
		min_row = 0 
		min_col = 0 
		for j, max_row in enumerate(self.max_rows):
			min_col = 0 
			for i, max_col in enumerate(self.max_cols):
				idx = j * col_len + i
				new_array[idx] = np.min(img_data[:,min_row:max_row,min_col:max_col])
				min_col = max_col
			min_row = max_row
		new_array = np.reshape(new_array, (self._dims,))
		
		observation_conversion = Vector(new_array)

		return observation_conversion