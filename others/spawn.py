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
				 dz=4, # this will spawn w/dz-meters above (positive) object (roof or floor)
				 yaw=0,
				 bounds_component=None,
				 random_yaw=True,
				 random=False,
				 vertical = True,
				 ):
		super().__init__()

	def connect(self):
		super().connect()
		# define if spawn method will be random or static
		if self.random:
			self.get_spawn = self.random_spawn
		else:
			self.get_spawn = self.static_spawn
			self._x = self.x
			self._y = self.y
			self._z = self._map.get_roof(self._x, self._y, self.dz)
			self._yaw = self.yaw

	# uniform distribution between passed in range
	def get_random_pos(self):
		if self.vertical:
			x, y, z = self._bounds.get_random()
			z = self._map.get_roof(x, y, self.dz)
		else:
			while (True):
				x, y, z = self._bounds.get_random()
				in_object = self._map.at_object_2d(x, y)
				if not in_object:
					break
		return x, y, z
	
	# generate random spawn until outside of an object
	def random_spawn(self):
		self._x, self._y, self._z = self.get_random_pos()
		if self.random_yaw:
			# make yaw face towards origin (with some noise)
			# this is used to make sure drone navigates through buildings (most of the time)
			#curr_position = np.array([self._x, self._y, self._z], dtype=float)
			#facing_position = np.array([0, 0, 0], dtype=float)
			#distance_vector = facing_position - curr_position
			#facing_yaw = math.atan2(distance_vector[1], distance_vector[0])
			#noise = np.random.normal(0, np.pi/6)
			#self._yaw = facing_yaw + noise
			self._yaw = np.random.uniform(-1*np.pi, np.pi)
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