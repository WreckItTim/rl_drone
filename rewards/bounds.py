# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math

class Bounds(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, 
                 drone_component, 
                 x_bounds,
                 y_bounds,
                 z_bounds,
	):
		super().__init__()

	# calculates rewards from agent's current state (call to when taking a step)
	def reward(self, state):
		_drone_position = self._drone.get_position()
		x = _drone_position[0]
		y = _drone_position[1]
		z = _drone_position[2]
		value = 0
		if x < self.x_bounds[0] or x > self.x_bounds[1]:
			value = -10
		if y < self.y_bounds[0] or y > self.y_bounds[1]:
			value = -10
		if z < self.z_bounds[0] or y > self.z_bounds[1]:
			value = -10
		return value