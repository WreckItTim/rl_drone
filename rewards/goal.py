from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math
import utils

# calculates distance between drone and point relative to starting position/orientation
class Goal(Reward):
	# constructor, set the relative point and min-max distances to normalize by
	@_init_wrapper
	def __init__(self,
				drone_component, 
				goal_component, 
			  	value_type='scale2', # see if statements in step() function
				tolerance=4, # min distance from goal for success 
				include_z=True,
				to_start=True,
				# if to_start=True will calculate rewards relative to start position
				# if to_start=False will calculate rewards relative to last position
				terminate=True, # =True will terminate episodes when Goal
		):
		super().__init__()
		#self.init_normalization()

	def get_distance(self):
		_drone_position = np.array(self._drone.get_position(), dtype=float)
		_goal_position = np.array(self._goal.get_position(), dtype=float)
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
			_goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
		distance_vector = _goal_position - _drone_position
		distance = np.linalg.norm(distance_vector)
		return distance
	
	# get reward based on distance to point 
	def step(self, state):
		distance = self.get_distance() + 1e-4
		d = max(1, distance / self._last_distance)

		if not self.to_start:
			self._last_distance = distance

		if self.value_type == 'exp':
			distance_reward = 2 * (math.exp(math.log(0.5)*d) - 0.5)
		if self.value_type == 'scale':
			distance_reward = -1*distance
		if self.value_type == 'scale2':
			distance_reward = -1*d

		done = False
		value = distance_reward
		if distance <= self.tolerance:
			value += 10
			done = True
			state['termination_reason'] = 'goal'
			state['termination_result'] = 'success'

		return value, done and self.terminate

	def reset(self, state):
		self._last_distance = self.get_distance() + 1e-4