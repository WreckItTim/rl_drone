# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from component import _init_wrapper
from sensors.sensor import Sensor
from observations.image import Image
from transformers.transformer import Transformer
import numpy as np
from gym import spaces

class Multi(Observer):
	
	# constructor
	@_init_wrapper
	def __init__(
		self,
		vector_observer_component,
		image_observer_component,
		):
		super().__init__()
		
	# gets observations
	def observe(self, write=False):
		# get observations
		vector_data, vector_name = self._vector_observer.observe(write)
		image_data, image_name = self._image_observer.observe(write)
		data_dict = {
			"vec": vector_data,
			"img": image_data,
			}
		return data_dict, 'Mutli_' + vector_name + '_' + image_name

	def reset(self):
		super().reset()
		self._vector_observer.reset()
		self._image_observer.reset()

	# returns dict space with proper dimensions
	def get_space(self):
		space_dict = spaces.Dict(
			spaces={
				"vec": self._vector_observer.get_space(),
				"img": self._image_observer.get_space(),
				}
			)
		return space_dict