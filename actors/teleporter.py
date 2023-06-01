from actors.actor import Actor
from component import _init_wrapper
from gym import spaces
import numpy as np
import rl_utils as utils
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
		self._last_state = [[0,0,0], 0]

	# backtracks to state before last action
	def undo(self):
		self._drone.teleport(self._last_state[0][0], self._last_state[0][1], self._last_state[0][2], self._last_state[1], ignore_collision=False)
		for idx, action in enumerate(self._actions):
			action.undo()

	# interpret action from RL
	def step(self, state):
		current_position = self._drone.get_position() # meters
		current_yaw = self._drone.get_yaw() # radians
		#print(current_position, current_yaw)
		self._last_state = [current_position, current_yaw]
		target = {
			'x':current_position[0],
			'y':current_position[1],
			'z':current_position[2],
			'yaw':current_yaw,
			}
		for idx, action in enumerate(self._actions):
			action._idx = idx # tell action which index it is
			this = action.step(state, execute=False) # transcribe action but do not take
			#print(this)
			for key in target:
				if key in this:
					target[key] += this[key]
		# teleport drone
		#print('teleport', target['x'], target['y'], target['z'], target['yaw'])
		self._drone.teleport(target['x'], target['y'], target['z'], target['yaw'], ignore_collision=False)

		
	# randomly sample RL output from action space unless specified
	def debug(self, state=None):
		sampled_output = state['rl_output']
		utils.speak(f'taking actions from continuous action-space: {sampled_output}...')
		self.step(state)