# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

class Steps(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			  value_type='scale2', # see if statements in step() function
			  ):
		super().__init__()

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		nSteps = state['nSteps']

		if self.value_type == 'constant':
			value = -1
		if self.value_type == 'scale':
			value = -1 * nSteps

		return value, False