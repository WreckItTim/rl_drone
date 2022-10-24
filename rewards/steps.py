# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper

class Steps(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, max_steps=100):
		super().__init__()
		self._nSteps = 0

	# calculates rewards from agent's current state (call to when taking a step)
	def reward(self, state):
		self._nSteps += 1
		value = 0 
		if self._nSteps >= self.max_steps:
			value -= 10
		return value

	def reset(self):
		self._nSteps = 0