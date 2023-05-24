# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

class MaxSteps(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			  max_steps, # initial number of max steps before episode termination (use update_steps to scale)
			  terminate=True, # =True will terminate episodes on collision
			  ):
		super().__init__()
		
	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		nSteps = state['nSteps']

		done = False
		value = 0
		if nSteps >= self.max_steps:
			done = True
			value = -1
		if done and self.terminate:
			state['termination_reason'] = 'steps'
			state['termination_result'] = 'failure'
		return value, done and self.terminate