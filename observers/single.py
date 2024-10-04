# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor
from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer
from gym import spaces
import numpy as np
import rl_utils as utils

class Single(Observer):
	
	# and observer with same observation types (can be multiple sensors)
	# observation type is either vector or image
	# image shapes are channel first, rows, cols
	@_init_wrapper
	def __init__(
		self, 
		sensors_components,
		vector_length = None,
		is_image = False,
		image_height = None, 
		image_width = None,
		image_bands = None,
		nTimesteps = 1
	):
		super().__init__(
		)
		if is_image:
			self._output_shape = (image_bands * nTimesteps, image_height, image_width)
			self._history = np.full(self._output_shape, 0, dtype=np.uint8)
		else:
			self._output_shape = (vector_length * nTimesteps,)
			self._history = np.full(self._output_shape, 0, dtype=float)
		self._old_names = []

	# gets observations
	def step(self, state=None):
		# make observations and stack into local image/vector
		next_array = []
		new_names = []
		for sensor in self._sensors:
			# get obeservation
			if sensor.offline:
				if self.is_image:
					empty_array = np.full((self.image_bands, self.image_height, self.image_width), 0, dtype=np.uint8)
					empty_name = 'I0'
				else:
					empty_array = np.full((self.vector_length,), 0, dtype=float)
					empty_name = 'V0'
				next_array.append(empty_array)
				new_names.append(empty_name)
			else:
				observation = sensor.step(state)
				this_array = observation.to_numpy()
				next_array.append(this_array)
				new_names.append(observation._name)
		# concatenate observations
		axis = 0
		array = np.concatenate(next_array, axis)
		if self.nTimesteps == 1:
			name = '_'.join(new_names)
			return array.copy(), name
		# rotate saved timesteps in history
		if self.is_image:
			for i in range(self.nTimesteps-1, 0, -1):
				start_i = i * self.image_bands
				save_to = slice(start_i, start_i + self.image_bands)
				start_i = (i-1) * self.image_bands
				load_from = slice(start_i, start_i + self.image_bands)
				self._history[save_to,:,:] = self._history[load_from,:,:]
			save_to = slice(0, self.image_bands)
			self._history[save_to,:,:] = array
		else:
			for i in range(self.nTimesteps-1, 0, -1):
				start_i = i * self.vector_length
				save_to = slice(start_i, start_i + self.vector_length)
				start_i = (i-1) * self.vector_length
				load_from = slice(start_i, start_i + self.vector_length)
				self._history[save_to] = self._history[load_from]
			save_to = slice(0, self.vector_length)
			self._history[save_to] = array
		self._old_names = [new_names] + self._old_names
		if len(self._old_names) > self.nTimesteps:
			self._old_names.pop(-1)
		name = '_'.join(sum(self._old_names, []))
		return self._history.copy(), name

	def reset(self, state=None):
		for sensor in self._sensors:
			sensor.reset(state)
		if self.is_image:
			self._history = np.full(self._output_shape, 0, dtype=np.uint8)
		else:
			self._history = np.full(self._output_shape, 0, dtype=float)
		self._old_names = []

	# returns box space with proper dimensions
	def get_space(self):
		if self.is_image:
			return spaces.Box(low=0, high=255, shape=self._output_shape, dtype=np.uint8)
		return spaces.Box(0, 1, shape=self._output_shape, dtype=float)
