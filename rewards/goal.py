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
				 min_distance=0, # goal tolerance 
				 max_distance=120, # how far can deviate from goal
				 include_z=True,
				 to_start=True,
				 # if to_start=True will calculate rewards relative to start position
				 # if to_start=False will calculate rewards relative to last position
				 ):
		super().__init__()
		#self.init_normalization()

	def get_distance(self):
		_drone_position = self._drone.get_position()
		_goal_position = self._goal.get_position()
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
			_goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
		distance_vector = _goal_position - _drone_position
		distance = np.linalg.norm(distance_vector)
		return distance
	
	# get reward based on distance to point 
	def reward(self, state):
		distance = self.get_distance()
		d = distance / self._last_distance

		if not self.to_start:
			self._last_distance = distance

		distance_reward = 2 * (math.exp(math.log(0.5)*d) - 0.5)

		value = distance_reward
		if distance <= self.min_distance:
			value += 10
		if distance >= self.max_distance:
			value -= 10

		return value

	def reset(self):
		self._last_distance = self.get_distance()

	'''
	# calculate constants for normalization
	def init_normalization(self):
		# normalize to min and max distances
		self._diff = self.max_distance - self.min_distance

	# normalize reward value between 0 and 1
	def normalize_reward(self, distance):
		# clip distance
		clipped_distance = max(self.min_distance, min(self.max_distance, distance))
		# normalize distance to fit desired behavior of reward function
		normalized_distance = (clipped_distance - self.min_distance) / self._diff
		# get value from reward function
		value = self._reward_function(normalized_distance)
		return value

	def get_yaw_diff(self):
		goal_yaw = utils.position_to_yaw(distance_vector)
		drone_yaw = self._drone.get_yaw()
		yaw_diff = math.pi - abs(abs(goal_yaw - drone_yaw) - math.pi)
		return yaw_diff
	'''