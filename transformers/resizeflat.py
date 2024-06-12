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
		self._width = len(self.max_rows)
		self._height = len(self.max_cols)
		self._mask = np.full((self._height, self._width), True)

	# if observation type is valid, applies transformation
	def transform(self, observation):
		img_data = observation.to_numpy()
		# TEMP CODE TODO:DELETE DELETE DELETE
		#img_data = np.moveaxis(img_data, 0, 2)
		#cv2.imwrite(utils.get_global_parameter('working_directory') + 'tello_imgs/' + observation._name + '_post.png', img_data)
		#img_data = np.moveaxis(img_data, 2, 0)
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
		new_array = np.reshape(self._mask.flatten() * new_array, (self._dims,))
		
		observation_conversion = Vector(new_array)

		return observation_conversion
	
	# will scale down number of flattened sensors to use
	def scale_to(self, level):
		# 0th level is forward distance
		# 1st level includes boundary
		# 2nd and subsequent levels split the differnce between previous levels
		# warning past level 2 sacles exponentially
		# example 9x9 grid showing levels 0-3
		# the sensor max resolution is level 3 - nodes are flipped off
		# all levels include previous levels as well
		'''
		1 1 1 1 1 1 1 1 1
		1 3 3 3 3 3 3 3 1
		1 3 2 2 2 2 2 3 1
		1 3 2 3 3 3 2 3 1
		1 3 2 3 0 3 2 3 1
		1 3 2 3 3 3 2 3 1
		1 3 2 2 2 2 2 3 1
		1 3 3 3 3 3 3 3 1
		1 1 1 1 1 1 1 1 1
		'''
		# this is done to keep the min and max observable space the same
		# while increasing the intermediate resolution values
		height = self._height
		width = self._width
		center_row = int(height/2)
		center_col = int(width/2)
		if width != height:
			print('WARNING: scale_to only works for squares')
		mask = np.full((width, height), False)
		# this helper function fills a square permiter at given distance from boundary
		def fill_rect(d):
			for r in range(height-d):
				mask[r, d] = True
				mask[r, width-1-d] = True
			for c in range(width-d):
				mask[d, c] = True
				mask[height-1-d, c] = True
		def fill_rect(d_height, d_width):
			d_height, d_width = int(d_height), int(d_width)
			for r in range(d_height, height-d_height):
				mask[r, d_height] = True
				mask[r, width-1-d_height] = True
			for c in range(d_width, width-d_width):
				mask[d_width, c] = True
				mask[height-1-d_width, c] = True
		# TODO this is hard coded a bit to a 9x9 with 0-4 levels, fix it to work for all
		# level 0 (off)
		# level 1 (center)
		if level >= 1:
			fill_rect(center_row, center_col)
		# level 2 (boundary)
		if level >= 2:
			fill_rect(0,0)
		# level 3+ (intermediates)
		if level >= 3:
			fill_rect(center_row/2, center_col/2)
		if level >= 4:
			fill_rect(center_row-(2-1),center_col-(2-1))
			fill_rect(center_row-(2+1),center_col-(2+1))
		self._mask = mask.copy()


	# reset to starting values
	def reset(self, state = None):
		self._mask = np.full((self._height, self._width), True)
