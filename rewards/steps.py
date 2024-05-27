# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

class Steps(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			  ):
		super().__init__()

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		nSteps = state['nSteps']

		value = -1

		return value, False