from actions.action import Action
from component import _init_wrapper
import math
import rl_utils as utils

# coutinuouts output will scale resolution of input sensors
# sensors should start at maximum scale
class Slim(Action):
	# constructor takes a list of components to scale down resolution
	@_init_wrapper
	def __init__(self, 
				slimmable_component, # list of components to call scale_down() on
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_slim = 0.125,
				min_space = 0, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				active = True, # False will just return default behavior
				discrete = False,
				):
		self._last_state = 1
		self._rho = 1

	def undo(self):
		self.set_slim(self._last_state)

	def set_slim(self, rho):
		self._rho = rho # give access to other components to last rho
		self._slimmable.set_slim(rho)
		
	def reset(self, state=None):
		self.set_slim(1.0)
		
	# move at input rate
	def step(self, state, execute=True):
		self._last_state = self._rho
		if not self.active:
			rho = 1
		else:
			rl_output = state['rl_output'][self._idx]
			if not self.discrete:
				rho = round(max(self.min_slim, rl_output), 4)
			else:
				idx = max(0, min(len(self.discrete)-1, int(rl_output * len(self.discrete))))
				rho = self.discrete[idx]
		self.set_slim(rho)
		#utils.speak(f'set slim:{rho}')
		return {'slim':rho}