
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

# rewards based on the memory size of split point
class Split(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			split_component, # get compaction-factor c from action
		):
		super().__init__()
	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		if 'split' in state:
			c = state['split']
		else:
			c = self._split._c
		
		value = -1 * c

		return value, False