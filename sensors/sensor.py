# abstract class used to handle sensors
from component import Component

class Sensor(Component):
	# constructor, offline is a boolean that handles if sensor goes offline
	def __init__(self,
			  offline,
			  ):
		pass

	# fetch a response from sensor
	def sense(self):
		raise NotImplementedError

	def transform(self, observation):
		if self._transformers is not None:
			for transformer in self._transformers:
				observation_conversion = transformer.transform(observation)
				if observation_conversion is not None:
					observation = observation_conversion
		return observation

	# when using the debug controller
	def debug(self):
		observation = self.sense()
		observation.display()

	def connect(self):
		super().connect()
