from actors.actor import Actor
from component import _init_wrapper
from gym import spaces
import numpy as np
import utils
import random
import math

# uses continuous actions to teleport (set yaw and position)
# this is much more quicker, precise, and stable than real-time movements
# suggested to add noise to simulate more realistic movements
class Teleporter(Actor):
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
		current_position = self._drone.get_position() # meters
		current_yaw = self._drone.get_yaw() # radians
		target = {
			'x':current_position[0],
			'y':current_position[1],
			'z':current_position[2],
			'yaw':current_yaw,
			}
		for idx, action in enumerate(self._actions):
			print('teleport step', idx)
			action._idx = idx # tell action which index it is
			this = action.step(state, execute=False) # transcribe action but do not take
			for key in target:
				if key in this:
					target[key] += this[key]
		# teleport drone
		print('Target:', target)
		self._drone.teleport(target['x'], target['y'], target['z'], target['yaw'], ignore_collision=False)
		print('Actual:', self._drone.get_position(), self._drone.get_yaw())
		print('Collided?', self._drone.check_collision())
		
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


	# returns continous action space of type Box
	# defined from action components' min and max space vars
	def get_space(self):
		nActions = len(self._actions)
		min_space = np.zeros(nActions)
		max_space = np.zeros(nActions)
		for idx in range(nActions):
			min_space[idx] = self._actions[idx].min_space
			max_space[idx] = self._actions[idx].max_space
		return spaces.Box(min_space, max_space)