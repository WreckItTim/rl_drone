from actors.actor import Actor
from component import _init_wrapper
from gymnasium import spaces
import numpy as np
import rl_utils as utils
import random
import math

# uses continuous actions to teleport (set yaw and position)
# this is much more quicker, precise, and stable than real-time movements
# suggested to add noise to simulate more realistic movements
class TeleporterDiscrete(Actor):
	# constructor
	@_init_wrapper
	def __init__(self,
	drone_component,
	actions_components=[],
	):
		super().__init__()
		self._type = 'continuous'


	# interpret action from RL
	def step(self, state):
		if 'transcribed_action' not in state:
			state['transcribed_action'] = {}
		current_position = self._drone.get_position() # meters
		current_yaw = self._drone.get_yaw() # radians
		target = {
			'x':current_position[0],
			'y':current_position[1],
			'z':current_position[2],
			'yaw':current_yaw,
			}
		rl_output = state['rl_output']
		this = self._actions[rl_output].step(state, execute=False) # transcribe action but do not take
		state['transcribed_action'].update(this)
		for key in target:
			if key in this:
				target[key] += this[key]
		# teleport drone
		state['transcribed_action'].update(target)
		if 'end' in state['transcribed_action']:
			if state['transcribed_action']['end']:
				state['termination_reason'] = state['transcribed_action']['end_reason']
				return True
		self._drone.teleport(target['x'], target['y'], target['z'], target['yaw'], ignore_collision=False)
		return False
		
	# randomly sample RL output from action space unless specified
	def debug(self, state=None):
		if state is None:
			x = utils.prompt('enter r to randomize or rl_output values')
			if x == 'r':
				sampled_output = np.zeros(len(self._actions), dtype=float)
				for idx, action in enumerate(self._actions):
					sampled_output[idx] = np.random.uniform(action.min_space, action.max_space, size=None)
			else:
				sampled_output = np.array([float(_) for _ in x.split(' ')])
			state = {
				'rl_output': sampled_output,
			}
		else:
			sampled_output = state['rl_output']
		utils.speak(f'taking actions from continuous action-space: {sampled_output}...')
		self.step(state)
		
	# returns dioscrete action space of type Discrete
	def get_space(self):
		return spaces.Discrete(len(self._actions))