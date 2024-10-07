# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from component import _init_wrapper
from sensors.sensor import Sensor
from observations.image import Image
from transformers.transformer import Transformer
import numpy as np
from gymnasium import spaces

class Multi(Observer):
	
	# constructor
	@_init_wrapper
	def __init__(
		self,
		vector_observer_component,
		image_observer_component,
		):
		super().__init__()

	def null_data(self):
		# get observations
		vector_data = self._vector_observer.null_data()
		# cleanup after getting last data fetches
		image_data = self._image_observer.null_data()
		data_dict = {
			"vec": vector_data,
			"img": image_data,
			}
		return data_dict
		
	# gets observations
	def step(self, state=None):
		# get observations
		vector_data, vector_name = self._vector_observer.step(state)
		# cleanup after getting last data fetches
		image_data, image_name = self._image_observer.step(state)
		data_dict = {
			"vec": vector_data,
			"img": image_data,
			}
		return data_dict, vector_name + '__' + image_name

	def reset(self, state=None):
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