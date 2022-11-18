# continuous actions - RL returns an index specifying which action to take
from actors.actor import Actor
from component import _init_wrapper
from gym import spaces
import numpy as np

class ContinuousActor(Actor):
	# constructor
	@_init_wrapper
	def __init__(self, actions_components=[]):
		super().__init__()

	# returns dioscrete action space
	def get_space(self):
		nActions = len(self._actions)
		min_vals = np.array([-1]*nActions)
		max_vals = np.array([1]*nActions)
		return spaces.Box(min_vals, max_vals)
		
	# interpret action from RL
	def act(self, rl_output):
		for idx in range(len(rl_output)):
			self._actions[idx].act(rl_output[idx])
			# for string output
			rl_output[idx] = round(rl_output[idx], 2)
		return str(rl_output)