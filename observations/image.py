# used to handle image observations saved as np arrays
from observations.observation import Observation
import cv2

class Image(Observation):

	# constructor
	def __init__(self, _data, data_path=None, is_gray=False):
		super().__init__(_data=_data, data_path=data_path)
		self.is_gray = is_gray

	# displays observation
	def display(self):
		cv2.imshow(f'Observation {self._name}:', self._data)
		cv2.waitKey(0)
		cv2.destroyAllWindows()