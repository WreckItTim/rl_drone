from actions.action import Action
from component import _init_wrapper
import math

# ends episode
class End(Action):
	@_init_wrapper
	def __init__(self, 
				zero_threshold=0.1, # absolute value of rl_output below this will do nothing (true zero)
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = -1, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				active = True, # False will just return default behavior
			):
		pass
		
	# move at input rate
	def step(self, state, execute=True):
		if not self.active:
			return {}
		rl_output = state['rl_output'][self._idx]
		if rl_output > self.zero_threshold:
			return {'end':True, 'end_reason':'action'}
		return {'end':False}