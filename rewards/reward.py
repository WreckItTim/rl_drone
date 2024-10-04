
from component import Component

class Reward(Component):
	# constructor
	def __init__(self):
		pass

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		raise NotImplementedError

	def connect(self):
		super().connect()