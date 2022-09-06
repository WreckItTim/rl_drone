# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor
from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer
from gym import spaces
import numpy as np

class Single(Observer):
	
	# and observer with same observation types (can be multiple sensors)
	# observation type is either vector or image
	@_init_wrapper
	def __init__(
		self, 
		sensors_components,
		vector_length = -1,
		is_image = False,
		image_height = None, 
		image_width = None,
		image_bands = None,
	):
		super().__init__(
		)
		if is_image:
			self._output_shape = (image_height, image_width, image_bands)
		else:
			self._output_shape = (vector_length,)
		
	# gets observations
	def observe(self, write=False):
		# make observations and stack into global image/vector
		arrays = []
		name = 'Single'
		for sensor in self._sensors:
			# get obeservation
			observation = sensor.sense()
			if write: 
				observation.write()
			arrays.append(observation.to_numpy())
			name += '_' + observation._name
		# concatenate observations
		axis = 2 if self.is_image else 0
		array = np.concatenate(arrays, axis)
		return array, name

	def reset(self):
		super().reset()
		for sensor in self._sensors:
			sensor.reset()

	# returns box space with proper dimensions
	def get_space(self):
		if self.is_image:
			return spaces.Box(0, 255, shape=self._output_shape, dtype=np.uint8)
		return spaces.Box(0, 1, shape=self._output_shape, dtype=np.float64)