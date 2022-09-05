# used to handle image observations saved as np arrays
import utils
from observations.observation import Observation

class Vector(Observation):

	# constructor
	def __init__(self, 
			  _data, 
			  names = None, 
			  data_path = None,
		   ):
		super().__init__(_data = _data, data_path = data_path)
		self.names = names

	# displays observation to console
	def display(self):
		utils.speak(self._data)

	# write as json
	def write(self, 
		   directory_path = None, 
		   file_name = None,
		   ):
		file_path = super().write(
			directory_path = directory_path, 
			file_name = file_name,
			)
		self.data_path = file_path + '.json'
		# create dict from data
		if self.names is None:
			self.names = [str(idx) for idx in range(len(self._data))]
		data_dict = {name:self._data[idx] for idx, name in enumerate(self.names)}
		utils.write_json(data_dict, self.data_path)