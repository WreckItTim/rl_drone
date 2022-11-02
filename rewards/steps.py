# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math

class Steps(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, max_steps=100):
		super().__init__()

	# calculates rewards from agent's current state (call to when taking a step)
	def reward(self, state):
		nSteps = state['nSteps']
		s = nSteps / self.max_steps
		value = 1-1/math.exp(math.log(0.5)*s**2)
		return value