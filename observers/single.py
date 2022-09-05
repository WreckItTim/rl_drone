# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from sensors.sensor import Sensor
from component import _init_wrapper
from sensors.sensor import Sensor
from transformers.transformer import Transformer
from gym import spaces
import numpy as np

class Single(Observer):
	
	# constructor
	@_init_wrapper
	def __init__(
		self, 
		sensor_component,
		vector_length = -1,
		vector_names = None,
		is_image = False,
		image_height = None, 
		image_width = None,
		image_bands = None,
		transformers_components = [],
	):
		super().__init__(
		)
		if is_image:
			self._output_shape = (image_height, image_width, image_bands)
		else:
			self._output_shape = (vector_length,)
		
	# gets observations
	def observe(self, write=False):
		# get obeservation
		observation = self._sensor.sense(logging_info = self.vector_names)
		# make any transformations
		for transformer in self._transformers:
			transformer.transform(observation)
		if write: 
			observation.write()
		return observation.to_numpy(), observation._name

	def reset(self):
		super().reset()
		self._sensor.reset()

	# returns box space with proper dimensions
	def get_space(self):
		if self.is_image:
			return spaces.Box(0, 255, shape=self._output_shape, dtype=np.uint8)
		return spaces.Box(0, 1, shape=self._output_shape, dtype=np.float64)