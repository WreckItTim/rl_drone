from others.other import Other
from component import _init_wrapper
import random
import numpy as np
import math
import rl_utils as utils

# data structure specifying a spawn zone
class Spawn(Other):
	# pass in either static x, y, z, yaw
	# or ranges for random values
	# constructor
	@_init_wrapper
	def __init__(self, 
				 map_component='Map',
				 x=0,
				 y=0,
				 z=0,
				 yaw=0,
				 bounds_component=None,
				 random_yaw = True,
				 random=False,
				 ):
		super().__init__()
		# define if spawn method will be random or static
		if random:
			self.get_spawn = self.random_spawn
		else:
			self.get_spawn = self.static_spawn
			self._x = x
			self._y = y
			self._z = z
			self._yaw = yaw

	# uniform distribution between passed in range
	def get_random_pos(self):
		x,y,z = self._bounds.get_random()
		in_object = self._map.at_object_2d(x, y)
		return x, y, z, in_object
	
	# generate random spawn until outside of an object
	def random_spawn(self):
		in_object = True
		while(in_object):
			self._x, self._y, self._z, in_object = self.get_random_pos()
		if self.random_yaw:
			# make yaw face towards origin (with some noise)
			# this is used to make sure drone navigates through buildings (most of the time)
			curr_position = np.array([self._x, self._y, self._z], dtype=float)
			facing_position = np.array([0, 0, 0], dtype=float)
			distance_vector = facing_position - curr_position
			facing_yaw = math.atan2(distance_vector[1], distance_vector[0])
			noise = np.random.normal(0, np.pi/6)
			self._yaw = facing_yaw + noise
		return [self._x, self._y, self._z], self._yaw
		
	# simply return a static spawn 
	def static_spawn(self):
		return [self._x, self._y, self._z], self._yaw

	# get the position of last spawn
	def get_position(self):
		return [self._x, self._y, self._z]
	
	# get the yaw of last spawn
	def get_yaw(self):
		return self._yaw

	# debug mode
	def debug(self):
		utils.speak('spawn = ' + str(self.get_spawn()))