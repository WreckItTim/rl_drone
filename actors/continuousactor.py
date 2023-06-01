from actors.actor import Actor
from component import _init_wrapper
import numpy as np
import rl_utils as utils
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
		self._last_state = [[0,0,0],0]
		
	# interpret action from RL
	def step(self, state):
		transcribed = {}
		for idx, action in enumerate(self._actions):
			action._idx = idx # tell action which index it is
			transcribed.update(action.step(state)) # take action
		state['transcribed_action'] = transcribed.copy()
		
	# randomly sample RL output from action space unless specified
	def debug(self, state=None):
		sampled_output = state['rl_output']
		utils.speak(f'taking actions from continuous action-space: {sampled_output}...')
		self.step(state)