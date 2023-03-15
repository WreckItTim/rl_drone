from others.other import Other
from component import _init_wrapper
import random
import math
import rl_utils as utils
import numpy as np
import copy

# set goal according to drone's starting position and orientation
class RelativeGoal(Other):

	@_init_wrapper
	def __init__(self, 
				drone_component, 
				map_component,
				bounds_component,
				static_r = 0, # relative distance for static goal from drone
				static_z = 0, # relative z for static goal from drone
				static_yaw = 0, # relative yaw for static goal from drone
				random_r = [0,0], # relative distance for random goal from drone
				random_z = [0,0], # relative z for random goal from drone
				random_yaw = [0,0], # relative yaw for random goal from drone
				random_point_on_train = False, # random goal when training?
				random_point_on_evaluate = False, # random goal when evaluating?
					# otherwise will default to static
				# these values are stored for amps (do not change) this is for file IO
				original_static_r = None,
				original_static_z = None,
				original_static_yaw = None,
				original_random_r = None,
				original_random_z = None,
				original_random_yaw = None,
			 ):
		if original_static_r is None:
			self.original_static_r = static_r
		if original_static_z is None:
			self.original_static_z = static_z
		if original_static_yaw is None:
			self.original_static_yaw = static_yaw
		if original_random_r is None:
			self.original_random_r = random_r.copy()
		if original_random_z is None:
			self.original_random_z = random_z.copy()
		if original_random_yaw is None:
			self.original_random_yaw = random_yaw.copy()
		
		self._x = static_r * np.cos(static_yaw)
		self._y = static_r * np.sin(static_yaw)
		self._z = static_z

	# resets any amps (start of new training loop)
	def reset_learning(self):
		self.static_r = self.original_static_r
		self.static_z = self.original_static_z
		self.static_yaw = self.original_static_yaw
		self.random_r = self.original_random_r.copy()
		self.random_z = self.original_random_z.copy()
		self.random_yaw = self.original_random_yaw.copy()

	def get_position(self):
		return [self._x, self._y, self._z]

	# gets absolute yaw (relative to 0 origin)
	def get_yaw(self):
		position = self.get_position()
		return math.atan2(position[1], position[0])
	
	def calculate_xyz(self, drone_position, relative_position, alpha):
		x0, y0, z0 = drone_position
		x1, y1, z1 = relative_position
		# rotate axis (alpha is a scalar length)
		x = x0 + alpha * x1 
		y = y0 + alpha * y1
		z = z0 + z1
		# check if goal is in an object
		in_object = self._map.at_object_2d(x, y)
		return x, y, z, in_object

	# need to recalculate relative point at each reset
	def reset(self, state):
		is_evaluation = state['is_evaluation_env']
		drone_position = self._drone.get_position()
		drone_yaw = self._drone.get_yaw()
		# random point?
		random_point = False
		if is_evaluation and self.random_point_on_evaluate:
			random_point = True
		elif not is_evaluation and self.random_point_on_train:
			random_point = True
		valid_point = True
		if random_point:
			# randomize until in bounds
			attempt = 0
			while(True):
				r = random.uniform(self.random_r[0], self.random_r[1])
				z = random.uniform(self.random_z[0], self.random_z[1])
				yaw = random.uniform(self.random_yaw[0], self.random_yaw[1])
				yaw = drone_yaw + yaw
				x = r * np.cos(yaw)
				y = r * np.sin(yaw)
				relative_position = [x, y, z]
				if self._bounds.check_bounds(drone_position[0]+x, drone_position[1]+y, drone_position[2]+z):
					break
				attempt += 1
				if attempt > 1000:
					utils.speak(f'ERR could not find goal in bounds at pos:{drone_position} and rel:{relative_position}')
					valid_point = False
					break
		else:
			yaw = drone_yaw + self.static_yaw
			x = self.static_r * np.cos(yaw)
			y = self.static_r * np.sin(yaw)
			z = self.static_z
			relative_position = [x, y, z]
		# shorten the distance until not inside of an object
		if valid_point:
			alpha = 1
			in_object = True
			while in_object:
				# if alpha gets too close then we need to do a different spawn
				if alpha < 0.1:
					utils.speak(f'ERR could not find goal outside of obj at pos:{drone_position} and rel:{relative_position}')
					valid_point = False
					break
				self._x, self._y, self._z, in_object = self.calculate_xyz(drone_position, relative_position, alpha)
				alpha -= 0.1
		if not valid_point:
				self.reset(state)