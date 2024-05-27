# used to handle image observations saved as np arrays
from observations.observation import Observation

class Vector(Observation):

	# constructor
	def __init__(self, 
			  _data, 
			  names = None, 
			  data_path = None,
		   ):
		super().__init__(_data = _data, data_path = data_path)
		# create dict from data
		self.names = names
		if names is None:
			self.names = [str(idx) for idx in range(len(self._data))]