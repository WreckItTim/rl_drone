# abstract class used to handle sensors
from component import Component

class Sensor(Component):
	# constructor
	def __init__(self):
		pass

	# fetch a response from sensor
	def sense(self, logging_info=None):
		raise NotImplementedError

	# when using the debug controller
	def debug(self):
		observation = self.sense()
		observation.display()

	def connect(self):
		super().connect()
