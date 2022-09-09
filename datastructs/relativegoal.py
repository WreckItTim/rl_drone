# abstract class used to handle abstract components
from datastructs.datastruct import DataStruct
from component import _init_wrapper
import random
import math
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
				 random_yaw_min = 0,
				 random_yaw_max = 2 * math.pi,
				 reset_on_step=False,
			 ):
		self.xyz_point = np.array(xyz_point, dtype=float)
		self._x = self.xyz_point[0]
		self._y = self.xyz_point[1]
		self._z = self.xyz_point[2]
	
	def get_xyz(self, position, yaw, alpha):
		x = position[0] + alpha*self._x * math.cos(yaw) + alpha*self._y * math.sin(yaw)
		y = position[1] + alpha*self._y * math.cos(yaw) + alpha*self._x * math.sin(yaw)
		z = position[2] + self._z
		in_object = self._map.at_object_2d(x, y)
		return x, y, z, in_object

	# need to recalculate relative point at each reset
	def reset(self):
		position = self._drone.get_position()
		if self.random_yaw:
			yaw = random.uniform(self.random_yaw_min, self.random_yaw_max)
		else:
			yaw = self._drone.get_yaw()  # yaw counterclockwise rotation about z-axis
		# shorten the distance until not in object (this is a cheap trick, better to think about points first)
		alpha = 1
		in_object = True
		while in_object:
			x, y, z, in_object = self.get_xyz(position, yaw, alpha)
			alpha -= 0.1
			if alpha < 0.1:
				utils.error('invalid objective point')
		self.xyz_point = np.array([x, y, z], dtype=float)

	def step(self, state):
		if self.reset_on_step:
			self.reset()
		state['goal'] = self.xyz_point.tolist()

	# when using the debug controller
	def debug(self):
		self.reset()
		print('Relative Goal:',  self.xyz_point)