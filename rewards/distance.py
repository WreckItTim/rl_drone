# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

class Distance(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, 
                 drone_component,
				 goal_component,
				 include_z=False,
				 max_distance=100,
	):
		super().__init__()

	def get_distance(self):
		_drone_position = np.array(self._drone.get_position(), dtype=float)
		_goal_position = np.array(self._goal.get_position(), dtype=float)
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
			_goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
		distance_vector = _goal_position - _drone_position
		distance = np.linalg.norm(distance_vector)
		return distance

	# calculates rewards from agent's current state (call to when taking a step)
	def reward(self, state):
		distance = self.get_distance()

		value = 0
		if distance > self.max_distance:
			value = -10
		return value