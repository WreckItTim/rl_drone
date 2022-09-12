# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor
from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer
from gym import spaces
import numpy as np
import pickle

class Single(Observer):
	
	# and observer with same observation types (can be multiple sensors)
	# observation type is either vector or image
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
			self._output_shape = (image_height, image_width, image_bands * nTimesteps)
		else:
			self._output_shape = (vector_length * nTimesteps,)
		self._history = np.zeros(self._output_shape)
		self._old_names = []
		
	# gets observations
	def observe(self, write=False):
		# make observations and stack into global image/vector
		next_array = []
		new_names = []
		for sensor in self._sensors:
			# get obeservation
			if sensor.offline:
				if self.is_image:
					empty_array = np.zeros((self.image_height, self.image_width, self.image_bands), dtype=np.uint8)
					empty_name = 'I0'
				else:
					empty_array = np.zeros((self.vector_length,), dtype=np.float64)
					empty_name = 'V0'
				next_array.append(empty_array)
				new_names.append(empty_name)
			else:
				observation = sensor.sense()
				if write: 
					observation.write()
				next_array.append(observation.to_numpy())
				new_names.append(observation._name)
		# concatenate observations
		axis = 2 if self.is_image else 0
		array = np.concatenate(next_array, axis)
		if self.nTimesteps == 1:
			return array, '_'.join(new_names)
		# rotate saved timesteps in history
		if self.is_image:
			for i in range(self.nTimesteps-1, 0, -1):
				start_i = i * self.image_bands
				save_to = slice(start_i, start_i + self.image_bands)
				start_i = (i-1) * self.image_bands
				load_from = slice(start_i, start_i + self.image_bands)
				self._history[:,:,save_to] = self._history[:,:,load_from]
			save_to = slice(0, self.image_bands)
			self._history[:,:,save_to] = array
		else:
			for i in range(self.nTimesteps-1, 0, -1):
				start_i = i * self.vector_length
				save_to = slice(start_i, start_i + self.vector_length)
				start_i = (i-1) * self.vector_length
				load_from = slice(start_i, start_i + self.vector_length)
				self._history[save_to] = self._history[load_from]
			save_to = slice(0, self.vector_length)
			self._history[save_to] = array
		name = '_'.join(self._old_names + new_names)
		self._old_names = new_names
		return self._history, name

	def reset(self):
		super().reset()
		for sensor in self._sensors:
			sensor.reset()

	# returns box space with proper dimensions
	def get_space(self):
		if self.is_image:
			return spaces.Box(0, 255, shape=self._output_shape, dtype=np.uint8)
		return spaces.Box(0, 1, shape=self._output_shape, dtype=np.float64)