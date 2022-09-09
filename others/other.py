# abstract class used to handle abstract components
from component import Component

class Other(Component):
	def __init__(self):
		raise NotImplementedError

	def step(self, state):
		pass

	def reset(self, reset_state):
		pass

	# when using the debug controller
	def debug(self):
		pass

	def connect(self):
		super().connect()