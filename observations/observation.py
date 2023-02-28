# abstract class used to handle observations - sensor responses
import rl_utils as utils
import os
import numpy as np

class Observation:
	nObservations = 0

	# constructor
	def __init__(self, _data, data_path=None):
		if data_path is not None:
			_data = self.read_data(data_path)
		self._data = np.array(_data)
		self.transformations = []
		Observation.nObservations += 1
		self._name = type(self).__name__[0] + str(Observation.nObservations)

	# converts observation into an nd numpy array 
	# some observation types are already numpy arrays
	def to_numpy(self):
		return np.array(self._data.copy(), dtype=float)

	# sets numpy array data
	def set_data(self, data):
		self._data = data.copy()

	# displays observation
	def display(self):
		print(f'Observation {self._name}:{self._data}')