from others.other import Other
from component import _init_wrapper
import numpy as np
import random

# data structure specifying valid bounds (use for various spawn/goal/etc checks)
# bounds are a hollow cylinder specified by inner and outter dimensions
# radius handles x,y and z handles z
#    |-------|
#   | x x x x |
#  | x |---| x |
# | x |     | x | ^^^
# | x |     | x | ^^^
#  | x |---| x |
#   | x x x x |
#    |-------|
class BoundsCircle(Other):
	@_init_wrapper
	def __init__(self, 
				center = [0,0,0],
				inner_radius = 0,
				outter_radius = 999,
				min_z = -999, # drone coords flip z
				max_z = 0,
				# to keep track of changes in learning
				original_center = None,
				original_inner_radius = None,
				original_outter_radius = None,
				original_min_z = None,
				original_max_z = None,
				):
		if original_center == None:
			self.original_center = center.copy()
		if original_inner_radius == None:
			self.original_inner_radius = inner_radius
		if original_outter_radius == None:
			self.original_outter_radius = outter_radius
		if original_min_z == None:
			self.original_min_z = min_z
		if original_max_z == None:
			self.original_max_z = max_z
		self.validate()

	def validate(self):
		if self.inner_radius > self.outter_radius:
			self.inner_radius = self.outter_radius
			self.outter_radius = self.inner_radius
		if self.min_z > self.max_z:
			self.min_z = self.max_z
			self.max_z = self.min_z

	# reset to original values
	def reset_learning(self):
		self.center = self.original_center.copy()
		self.inner_radius = self.original_inner_radius
		self.outter_radius = self.original_outter_radius
		self.min_z = self.original_min_z
		self.max_z = self.original_max_z
		self.validate()

	# returns True/False if within/outside bounds of hollow cylinder
	def check_bounds(self, x, y, z):
		self.validate()
		x = x - self.center[0]
		y = y - self.center[1]
		z = z - self.center[2]
		r = np.sqrt(x**2 + y**2)
		if r >= self.inner_radius and r >= self.inner_radius and z >= self.min_z:
			if r <= self.outter_radius and r <= self.outter_radius and z <= self.max_z:
				return True
		return False

	# gets random x,y,z point in bounds of hollow cylinder
	def get_random(self):
		self.validate()
		r = self.inner_radius
		theta = np.random.uniform(0, 2*np.pi)
		if self.inner_radius != self.outter_radius:
			mu = (self.inner_radius + self.outter_radius)/2
			sigma = (self.outter_radius - self.inner_radius) / 6
			r = min(self.outter_radius, max(self.inner_radius, 
												np.random.normal(mu, sigma)))
		x = r * np.cos(theta)
		y = r * np.sin(theta)
		z = self.min_z
		if self.min_z != self.max_z:
			mu = (self.min_z + self.max_z)/2
			sigma = abs(self.min_z - self.max_z) / 6
			z = min(self.max_z, max(self.min_z, 
											np.random.normal(mu, sigma)))
		x = x + self.center[0]
		y = y + self.center[1]
		z = z + self.center[2]
		return x,y,z