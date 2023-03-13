from others.other import Other
from component import _init_wrapper
import numpy as np
import random

# data structure specifying valid bounds (use for various spawn/goal/etc checks)
# bounds are cube specified by inner and outter dimensions
class Bounds(Other):
	@_init_wrapper
	def __init__(self, 
					inner_x = [0,0],
					outter_x = [-999,999],
					inner_y = [0,0],
					outter_y = [-999,999],
					inner_z = [0,0],
					outter_z = [-999,999],
					original_inner_x = None,
					original_outter_x = None,
					original_inner_y = None,
					original_outter_y = None,
					original_inner_z = None,
					original_outter_z = None,
				 ):
		super().__init__()
		if original_inner_x == None:
			self.original_inner_x = inner_x.copy()
		if original_outter_x == None:
			self.original_outter_x = outter_x.copy()
		if original_inner_y == None:
			self.original_inner_y = inner_y.copy()
		if original_outter_y == None:
			self.original_outter_y = outter_y.copy()
		if original_inner_z == None:
			self.original_inner_z = inner_z.copy()
		if original_outter_z == None:
			self.original_outter_z = outter_z.copy()
		self.validate_sides()
	
	# reset to original values
	def reset_learning(self):
		self.inner_x = self.original_inner_x.copy()
		self.outter_x = self.original_outter_x.copy()
		self.inner_y = self.original_inner_y.copy()
		self.outter_y = self.original_outter_y.copy()
		self.inner_z = self.original_inner_z.copy()
		self.outter_z = self.original_outter_z.copy()
		
	# check sides to make sure valid inner and outter dimensions
	# also see if either side has shrunk to zero
	def validate_sides(self):
		
		self._x_sides = []
		if self.inner_x[0] > self.inner_x[1]:
			mid_point = (self.inner_x[0] + self.inner_x[1])/2
			self.inner_x[0] = mid_point
			self.inner_x[1] = mid_point
		if self.outter_x[0] > self.inner_x[0]:
			self.outter_x[0] = self.inner_x[0]
		if self.outter_x[1] < self.inner_x[1]:
			self.outter_x[1] = self.inner_x[1]
		if self.inner_x[0] != self.outter_x[0]:
			self._x_sides.append(0)
		if self.inner_x[1] != self.outter_x[1]:
			self._x_sides.append(1)
			
		self._y_sides = []
		if self.inner_y[0] > self.inner_y[1]:
			mid_point = (self.inner_y[0] + self.inner_y[1])/2
			self.inner_y[0] = mid_point
			self.inner_y[1] = mid_point
		if self.outter_y[0] > self.inner_y[0]:
			self.outter_y[0] = self.inner_y[0]
		if self.outter_y[1] < self.inner_y[1]:
			self.outter_y[1] = self.inner_y[1]
		if self.inner_y[0] != self.outter_y[0]:
			self._y_sides.append(0)
		if self.inner_y[1] != self.outter_y[1]:
			self._y_sides.append(1)
			
		self._z_sides = []
		if self.inner_z[0] > self.inner_z[1]:
			mid_point = (self.inner_z[0] + self.inner_z[1])/2
			self.inner_z[0] = mid_point
			self.inner_z[1] = mid_point
		if self.outter_z[0] > self.inner_z[0]:
			self.outter_z[0] = self.inner_z[0]
		if self.outter_z[1] < self.inner_z[1]:
			self.outter_z[1] = self.inner_z[1]
		if self.inner_z[0] != self.outter_z[0]:
			self._z_sides.append(0)
		if self.inner_z[1] != self.outter_z[1]:
			self._z_sides.append(1)

	# increase/decrease bounds (see implemenation below)
	def apply_delta_inner(self, xyz):
		self.inner_x[0] += xyz[0]
		self.inner_x[1] -= xyz[0]
		self.inner_y[0] += xyz[1]
		self.inner_y[1] -= xyz[1]
		self.inner_z[0] += xyz[2]
		self.inner_z[1] -= xyz[2]
		self.validate_sides()

	# increase/decrease bounds (see implemenation below)
	def apply_delta_outter(self, xyz):
		self.outter_x[0] -= xyz[0]
		self.outter_x[1] += xyz[0]
		self.outter_y[0] -= xyz[1]
		self.outter_y[1] += xyz[1]
		self.outter_z[0] -= xyz[2]
		self.outter_z[1] += xyz[2]
		self.validate_sides()

	# returns True/False if within/outside bounds cube
	def check_bounds(self, x, y, z):
		if ( 
			((x <= self.inner_x[0] and x >= self.outter_x[0])
			or
			(x >= self.inner_x[1] and x <= self.outter_x[1]))
			and
			((y <= self.inner_y[0] and y >= self.outter_y[0])
			or
			(y >= self.inner_y[1] and y <= self.outter_y[1]))
			and
			((z <= self.inner_z[0] and z >= self.outter_z[0])
			or
			(z >= self.inner_z[1] and z <= self.outter_z[1]))
		):
			return True
		return False

	# gets random x,y,z point in bounds cube
	def get_random(self):
		x = self.inner_x[0]
		if len(self._x_sides) > 0:
			while(True):
				x_side = random.choice(self._x_sides)
				mu = (self.inner_x[x_side] + self.outter_x[x_side])/2
				sigma = abs(self.inner_x[x_side] - self.outter_x[x_side]) / 6
				x = np.random.normal(mu, sigma)
				# force inbounds
				if ((x <= self.inner_x[0] and x >= self.outter_x[0]) or
					(x >= self.inner_x[1] and x <= self.outter_x[1])):
					break
		y = self.inner_y[0]
		if len(self._y_sides) > 0:
			while(True):
				y_side = random.choice(self._y_sides)
				mu = (self.inner_y[y_side] + self.outter_y[y_side])/2
				sigma = abs(self.inner_y[y_side] - self.outter_y[y_side]) / 6
				y = np.random.normal(mu, sigma)
				# force inbounds
				if ((y <= self.inner_y[0] and y >= self.outter_y[0]) or
					(y >= self.inner_y[1] and y <= self.outter_y[1])):
					break
		z = self.inner_z[0]
		if len(self._z_sides) > 0:
			while(True):
				z_side = random.choice(self._z_sides)
				mu = (self.inner_z[z_side] + self.outter_z[z_side])/2
				sigma = abs(self.inner_z[z_side] - self.outter_z[z_side]) / 6
				z = np.random.normal(mu, sigma)
				# force inbounds
				if ((z <= self.inner_z[0] and z >= self.outter_z[0]) or
					(z >= self.inner_z[1] and z <= self.outter_z[1])):
					break
		return x,y,z