from others.other import Other
from component import _init_wrapper
import numpy as np
import random

# data structure specifying valid bounds (use for various spawn/goal/etc checks)
# bounds are a cube 
class BoundsCube(Other):
	@_init_wrapper
	def __init__(self, 
				center = [0,0,0],
				x = [0,0],
				y = [0,0],
				z = [0,0],
				):
		pass

	# returns True/False if within/outside bounds, ignoring z
	def check_bounds2D(self, x, y):
		x = x - self.center[0]
		y = y - self.center[1]
		if x >= self.x[0] and x <= self.x[1]:
			if y >= self.y[0] and y <= self.y[1]:
				return True
		return False

	# returns True/False if within/outside bounds
	def check_bounds(self, x, y, z):
		x = x - self.center[0]
		y = y - self.center[1]
		z = z - self.center[2]
		if x >= self.x[0] and x <= self.x[1]:
			if y >= self.y[0] and y <= self.y[1]:
				if z >= self.z[0] and z <= self.z[1]:
					return True
		return False

	# gets random x,y,z point in bounds
	def get_random(self):
		x = np.random.uniform(self.x[0], self.x[1])
		y = np.random.uniform(self.y[0], self.y[1])
		z = np.random.uniform(self.z[0], self.z[1])
		x = x + self.center[0]
		y = y + self.center[1]
		z = z + self.center[2]
		return x,y,z