from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math
import rl_utils as utils

# calculates distance between drone and point relative to starting position/orientation
class Distance(Reward):
	# constructor, set the relative point and min-max distances to normalize by
	@_init_wrapper
	def __init__(self,
				drone_component, 
				goal_component, 
			  	value_type='scale2', # see if statements in step() function
				max_distance = 100, # scale in meters
				include_z=True,
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
		this_distance = self.get_distance()
		delta_distance = self._last_distance - this_distance

		d = delta_distance / self.max_distance

		if self.value_type == 'exp':
			value = 2 * (math.exp(math.log(0.5)*d) - 0.5)
		if self.value_type == 'scale':
			value = delta_distance
		if self.value_type == 'scale2':
			value = d

		self._last_distance = this_distance
		return value, False

	def reset(self, state):
		self._last_distance = self.get_distance()