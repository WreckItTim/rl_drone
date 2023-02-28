
from component import Component
import rl_utils as utils

# abstract class used to handle sensors
class Sensor(Component):
	# constructor, offline is a boolean that handles if sensor goes offline
	def __init__(self,
			  offline,
			  ):
		pass

	# creates an observation object given a data array
	def create_obj(self, data):
		raise NotImplementedError

	# applies transformations on observation object
	def transform(self, observation):
		if self._transformers is not None:
			for transformer in self._transformers:
				observation_conversion = transformer.transform(observation)
				if observation_conversion is not None:
					observation = observation_conversion
		return observation

	def debug(self, state=None):
		observation = self.step(state)
		observation.display()

	def connect(self, state=None):
		super().connect()
