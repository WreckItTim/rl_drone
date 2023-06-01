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
				model_component, # list of components to call scale_down() on
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_slim = 0.125,
				active = True, # False will just return default behavior
			):
		self._last_state = 1

	def undo(self):
		self.set_rho(self._last_state)

	def set_rho(self, rho):
		self._rho = rho # give access to other components to last rho
		self._model._sb3model.slim = rho
		for module in self._model._actor.modules():
			#print(type(module))
			if 'Slim' in str(type(module)):
				module.slim = rho
		
	# move at input rate
	def step(self, state, execute=True):
		self._last_state = self._rho
		if not self.active:
			rho = 1
		else:
			rl_output = state['rl_output'][self._idx]
			rho = round(max(self.min_slim, rl_output), 4)
		self.set_rho(rho)
		#utils.speak(f'set slim:{rho}')
		return {'slim':rho}