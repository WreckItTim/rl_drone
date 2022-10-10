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
				 min_distance, 
				 max_distance, 
				 goal_tolerance=0,
				 include_z=True,
				 ):
		super().__init__()
		# set reward function
		#self._reward_function = lambda x : math.exp(-2.0 * x)
		#self._reward_function = lambda x : 1-x
		#self._reward_function = lambda x : 1-x
		self.init_normalization()

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
	
	# get reward based on distance to point 
	def reward(self, state):
		_drone_position = self._drone.get_position()
		_goal_position = self._goal.get_position()
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
			_goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
		distance_vector = _goal_position - _drone_position
		distance = np.linalg.norm(distance_vector)
		distance_reward = -0.5 * distance / self.max_distance

		goal_yaw = utils.position_to_yaw(distance_vector)
		drone_yaw = self._drone.get_yaw()
		yaw_to_goal = math.pi - abs(abs(goal_yaw - drone_yaw) - math.pi)
		yaw_reward = -0.5 * yaw_to_goal / math.pi

		value = yaw_reward + distance_reward
		if distance <= self.goal_tolerance:
			value += 10
		elif distance >= self.max_distance:
			value -= 10

		state['yaw_reward'] = yaw_reward
		state['pos_reward'] = distance_reward
		#print('distance:', utils._round(distance), 'angle:', utils._round(yaw_to_goal), 'reward:', value)
		return value

		#value = self.normalize_reward(distance)
		#return value 