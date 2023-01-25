from actors.actor import Actor
from component import _init_wrapper
from gym import spaces
import numpy as np
import utils
import random

# handles  continuous actions - RL returns an index specifying which action to take
class ContinuousActor(Actor):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  actions_components=[],
			  ):
		super().__init__()
		self._type = 'continuous'
		
	# interpret action from RL
	def step(self, state):
		for idx, action in enumerate(self._actions):
			action._idx = idx # tell action which index it is
			action.step(state) # take action
		state['transcribed_action'] = str(np.around(state['rl_output'], decimals=4))
		
	# randomly sample RL output from action space unless specified
	def debug(self, state=None):
		if state is None:
			sampled_output = np.zeros(len(self._actions), dtype=float)
			for idx, action in enumerate(self._actions):
				sampled_output[idx] = np.random.uniform(action._min_space, action._max_space, size=None)
			state = {
				'rl_output': sampled_output,
			}
		else:
			sampled_output = state['rl_output']
		utils.speak(f'taking actions from continuous action-space: {sampled_output}...')
		self.step(state)


	# returns continous action space of type Box
	# defined from action components' min and max space vars
	def get_space(self):
		nActions = len(self._actions)
		min_space = np.zeros(nActions)
		max_space = np.zeros(nActions)
		for idx in range(nActions):
			min_space[idx] = self._actions[idx]._min_space
			max_space[idx] = self._actions[idx]._max_space
		return spaces.Box(min_space, max_space)