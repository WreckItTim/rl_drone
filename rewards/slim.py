
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

# rewards based on the level of resolution
class Slim(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			slim_component, # get slim-factor rho from action
		):
		super().__init__()
	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		if 'slim' in state:
			rho = state['slim']
		else:
			rho = self._slim._rho
		
		value = -1 * rho

		return value, False