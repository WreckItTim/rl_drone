# used to handle image observations saved as np arrays
from observations.observation import Observation
import matplotlib.pyplot as plt
import numpy as np

class Image(Observation):

	# constructor
	def __init__(self, _data, data_path=None, is_gray=False):
		super().__init__(_data=_data, data_path=data_path)
		self.is_gray = is_gray

	# displays observation
	def display(self):
		# flip data to chanel last
		img = np.moveaxis(self._data, 0, 2)
		if self.is_gray:
			plt.imshow(img, cmap='gray', vmin=0, vmax=1)
		else:
			# convert BGR to RGB
			temp = img[:, :, 0].copy()
			img[:, :, 0] = img[:, :, 2].copy()
			img[:, :, 2] = temp.copy()
			plt.imshow(img)
		plt.show()