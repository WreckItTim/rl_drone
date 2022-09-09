# abstract class used to handle rewards in an RL enivornment
from component import Component

class Reward(Component):
	# constructor
	def __init__(self):
		pass

	# calculates rewards from agent's current state (call to when taking a step)
	def reward(self, state):
		raise NotImplementedError

	# when using the debug controller
	def debug(self):
		return self.reward({})

	def connect(self):
		super().connect()

	def reset(self, reset_state):
		pass