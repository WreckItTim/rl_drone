from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math
import rl_utils as utils

# calculates distance between drone and point relative to starting position/orientation
class Distance(Reward):
	@_init_wrapper
	def __init__(self,
				drone_component, 
				goal_component, 
				max_distance=8, # largest movement drone can make
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
		delta_distance = (self._last_distance - this_distance)/self.max_distance
		
		d = delta_distance
		value = np.tanh(d)

		self._last_distance = this_distance
		return value, False

	def reset(self, state):
		self._last_distance = self.get_distance()
