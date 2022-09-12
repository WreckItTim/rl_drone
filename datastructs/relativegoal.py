# abstract class used to handle abstract components
from datastructs.datastruct import DataStruct
from component import _init_wrapper
import random
import math
import utils
import numpy as np

# set goal according to drone's starting position and orientation
# WARNING: you must add this to the others_components in your environment
class RelativeGoal(DataStruct):

	@_init_wrapper
	def __init__(self, 
				 drone_component, 
				 map_component,
				 xyz_point,
				 random_yaw = False,
				 random_yaw_min = -1 * math.pi,
				 random_yaw_max = math.pi,
				 reset_on_step=False,
			 ):
		self.xyz_point = np.array(xyz_point, dtype=float)
		self._x = self.xyz_point[0]
		self._y = self.xyz_point[1]
		self._z = self.xyz_point[2]

	def get_position(self):
		return [self._x, self._y, self._z]

	# gets absolute yaw (relative to 0 origin)
	def get_yaw(self):
		return utils.position_to_yaw(self.get_position())
	
	def calculate_xyz(self, position, yaw, alpha):
		x = position[0] + alpha*self.xyz_point[0] * math.cos(yaw) + alpha*self.xyz_point[1] * math.sin(yaw)
		y = position[1] + alpha*self.xyz_point[1] * math.cos(yaw) + alpha*self.xyz_point[0] * math.sin(yaw)
		z = position[2] + self.xyz_point[2]
		in_object = self._map.at_object_2d(x, y)
		return x, y, z, in_object

	# need to recalculate relative point at each reset
	def reset(self):
		drone_position = self._drone.get_position()
		if self.random_yaw:
			relative_yaw = random.uniform(self.random_yaw_min, self.random_yaw_max)
		else:
			relative_yaw = self._drone.get_yaw()  # yaw counterclockwise rotation about z-axis
		# shorten the distance until not in object
		alpha = 1
		in_object = True
		while in_object:
			if alpha < 0.1:
				utils.error('invalid objective point')
			self._x, self._y, self._z, in_object = self.calculate_xyz(drone_position, relative_yaw, alpha)
			alpha -= 0.1
		print('goal:', utils._round(self.get_position()), utils._round(self.get_yaw()))

	# if reset on each step
	def step(self, state):
		if self.reset_on_step:
			self.reset()
		state['goal_position'] = self.get_position()
		state['goal_yaw'] = self.get_yaw()

	# when using the debug controller
	def debug(self):
		self.reset()