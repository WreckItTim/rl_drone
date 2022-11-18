# resizes image as flattened array, converts Image to Vector
from transformers.transformer import Transformer
from observations.vector import Vector
from skimage.transform import resize
import numpy as np
from component import _init_wrapper

class ResizeFlat(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self,
				length=5, # splits into this many columned sections
				max_rows=[42], # use only values above given row (gets rid of floor)
	):
		super().__init__()
		self._dims = length * len(max_rows)

	# if observation type is valid, applies transformation
	def transform(self, observation):
		#observation.check(Vector)
		# get max pixel value at each column section
		img_data = observation.to_numpy()
		nCols = img_data.shape[2]
		delta = int(nCols / self.length)

		new_array = np.zeros(self._dims, dtype=float)
		min_row = 0 
		for j, max_row in enumerate(self.max_rows):
			for i in range(self.length):
				idx = j * self.length + i
				if i == self.length - 1:
					new_array[idx] = np.min(img_data[:,min_row:max_row,i*delta:])
				else:
					new_array[idx] = np.min(img_data[:,min_row:max_row,i*delta:i*delta + delta])
			min_row += max_row
		new_array = np.reshape(new_array, (self._dims,))
		
		observation_conversion = Vector(new_array)

		return observation_conversion