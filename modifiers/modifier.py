# abstract class used to handle modifiers
# apply to any component and parent method
from component import Component

class Modifier(Component):
	def __init__(self,
			  modified_component,
		parent_method = 'reset',
		exectue = 'after',
		frequency = 1,
		counter = 0,
	):
		pass

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