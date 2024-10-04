from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math
import rl_utils as utils

# calculates orientation between drone and point relative to starting position/orientation
class Orientation(Reward):
	@_init_wrapper
	def __init__(self,
				drone_component, 
				goal_component, 
				include_z=True,
		):
		super().__init__()
		#self.init_normalization()

	def get_orientation(self):
		_drone_position = np.array(self._drone.get_position(), dtype=float)
		_goal_position = np.array(self._goal.get_position(), dtype=float)
		distance_vector = _goal_position - _drone_position
		yaw_1_2 = math.atan2(distance_vector[1], distance_vector[0])
		yaw1 = self._drone.get_yaw()
		yaw_diff = yaw_1_2 - yaw1
		if yaw_diff > math.pi:
			yaw_diff -= 2*math.pi
		if yaw_diff < -1*math.pi:
			yaw_diff += 2*math.pi
		return yaw_diff
	
	# get reward based on distance to point 
	def step(self, state):
		this_orientation = self.get_orientation()
		delta_orientation = self._last_orientation - this_orientation

		#o = np.abs(delta_orientation)
		o = np.abs(this_orientation)
		
		value = np.tanh(-1*o)

		self._last_orientation = this_orientation
		return value, False

	def reset(self, state):
		self._last_orientation = self.get_orientation()