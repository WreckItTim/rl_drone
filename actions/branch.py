from actions.action import Action
from component import _init_wrapper
import math
import rl_utils as utils

# coutinuouts output will select model from several branches
class Branch(Action):
	@_init_wrapper
	def __init__(self, 
				splits_components, # list of components to read model at given path
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				c_map = [], # maps int c to model path to read (increasing c will be penalized)
				min_space = -1, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				active = True, # False will just return default behavior
				max_c = -1, # max branch index, will use this by default (heaviest penalty)
			):
		if self.max_c == -1:
			self.max_c = len(self.c_map)-1
		
	# move at input rate
	def step(self, state, execute=True):
		if not self.active:
			c = self.max_c
		else:
			rl_output = state['rl_output'][self._idx]
			c = max(0, min(int((self.max_c+1) * rl_output), self.max_c))
		self._c = c # give access to other components to last branch index
		# tell all components to read model at index c
		for s_idx, split in enumerate(self._splits):
			split.read_model(self.c_map[s_idx][c])
		return {'split':c}