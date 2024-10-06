# used to handle image observations saved as np arrays
from observations.observation import Observation
import rl_utils as utils

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
			
	# displays observation
	def display(self):
	    for name_idx, name in enumerate(self.names):
	        print(name, '=', self._data[name_idx])
		
	# write data to file as a json
	def write(self, path_without_ftype, ftype='.json'):
	    data_dict = {}
	    list_data = self._data.to_list()
	    for name_idx, name in enumerate(self.names):
	        data_dict[name] = list_data[name_idx]
        write_json(data_dict, path_without_ftype+'.json'):
