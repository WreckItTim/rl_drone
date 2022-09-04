# abstract class used to handle observations - sensor responses
import utils
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
		self._name = 'Observation_' + str(Observation.nObservations)

	# writes observation metadata to given dir path
	def write(self, directory_path=None, file_name=None):
		if directory_path is None:
			directory_path = utils.get_global_parameter('working_directory') + '/observations/'
		if file_name is None:
			file_name = self._name
		file_path = directory_path + '/' + file_name
		if not os.path.exists(directory_path):
			os.mkdir(directory_path)
		#serialized = self._to_json()
		#utils.write_json(serialized, file_path + '.json')
		return file_path

	# when using the debug controller
	def debug(self):
		self.display()

	# displays observation to console
	def display(self):
		raise NotImplementedError

	# converts observation into an nd numpy array
	def to_numpy(self):
		return self._data

	# sets numpy array data
	def set_data(self, data):
		self._data = data

	# saves transformation info (after transformation is done)
	def save_transformation(self, transformer, data):
		#self.transformations.append(transformer._to_json())
		self.set_data(data)

	# turns observation in numpy array to be stacked with other arrays
	def to_stack(self, output_height, output_width):
		return self.to_numpy()