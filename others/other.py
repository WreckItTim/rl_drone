# abstract class used to misc components
from component import Component

class Other(Component):
	def __init__(self):
		raise NotImplementedError

	def step(self, state):
		pass

	def reset(self, reset_state):
		pass

	def reset_learning(self, reset_state):
		pass

	def save(self):
		pass

	def load(self):
		pass

	def debug(self):
		pass

	def connect(self):
		super().connect()

	def disconnect(self):
		super().disconnect()