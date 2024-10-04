
from component import Component

class Rewarder(Component):
	# constructor
	def __init__(self):
		pass

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		raise NotImplementedError

	def connect(self):
		super().connect()

	def reset(self, state=None):
		for reward in self._rewards:
			reward.reset(state)