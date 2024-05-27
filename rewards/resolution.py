
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

# rewards based on the level of resolution
class Resolution(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			resolution_component, # get level of resolution from action
			value_type = 'scale2', 
		):
		super().__init__()
	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		level = self._resolution._level
		max_level = self._resolution.max_level
		l = level / max_level

		value = -1 * l

		return value, False