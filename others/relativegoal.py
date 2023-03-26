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
				static_dz = 0, # relative z for static goal from drone (this is dz above roof or floor)
				static_yaw = 0, # relative yaw for static goal from drone
				random_r = [0,0], # relative distance for random goal from drone
				random_dz = [0,0], # relative z for random goal from drone (this is dz above roof or floor)
				random_yaw = [0,0], # relative yaw for random goal from drone
				random_point_on_train = False, # random goal when training?
				random_point_on_evaluate = False, # random goal when evaluating?
					# otherwise will default to static
				# these values are stored for amps (do not change) this is for file IO
				original_static_r = None,
				original_static_dz = None,
				original_static_yaw = None,
				original_random_r = None,
				original_random_dz = None,
				original_random_yaw = None,
				vertical = True,
			 ):
		if original_static_r is None:
			self.original_static_r = static_r
		if original_static_dz is None:
			self.original_static_dz = static_dz
		if original_static_yaw is None:
			self.original_static_yaw = static_yaw
		if original_random_r is None:
			self.original_random_r = random_r.copy()
		if original_random_dz is None:
			self.original_random_dz = random_dz.copy()
		if original_random_yaw is None:
			self.original_random_yaw = random_yaw.copy()
		
		self._x = static_r * np.cos(static_yaw)
		self._y = static_r * np.sin(static_yaw)
		self._z = static_dz

	# resets any amps (start of new training loop)
	def reset_learning(self):
		self.static_r = self.original_static_r
		self.static_dz = self.original_static_dz
		self.static_yaw = self.original_static_yaw
		self.random_r = self.original_random_r.copy()
		self.random_dz = self.original_random_dz.copy()
		self.random_yaw = self.original_random_yaw.copy()

	def set_position(self, x, y, z):
		self._x = x 
		self._y = y
		self._z = z

	def get_position(self):
		return [self._x, self._y, self._z]

	# gets absolute yaw (relative to 0 origin)
	def get_yaw(self):
		position = self.get_position()
		return math.atan2(position[1], position[0])

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
				dz = random.uniform(self.random_dz[0], self.random_dz[1])
				yaw = random.uniform(self.random_yaw[0], self.random_yaw[1])
				dx = r * np.cos(yaw)
				dy = r * np.sin(yaw)
				x = drone_position[0] + dx
				y = drone_position[1] + dy
				if self.vertical:
					z = self._map.get_roof(x, y, dz)
				else:
					z = -1*dz
				goal_position = np.array([x, y, z])
				in_object = False
				if not self.vertical:
					in_object = self._map.at_object_2d(x, y)
				
				if not in_object and self._bounds.check_bounds(x, y, z):
					break
				attempt += 1
				if attempt > 100:
					scale_x = 0.1*x
					scale_y = 0.1*y
					while(True):
						in_object = self._map.at_object_2d(x, y)
						if not in_object:
							break
						x -= scale_x
						y -= scale_y
					break
				if attempt > 1000:
					utils.speak(f'ERR could not find goal in bounds at drone:{drone_position} and goal:{goal_position}')
					valid_point = False
					break
		else:
			yaw = drone_yaw + self.static_yaw
			dx = self.static_r * np.cos(yaw)
			dy = self.static_r * np.sin(yaw)
			x = drone_position[0] + dx
			y = drone_position[1] + dy
			if self.vertical:
				z = self._map.get_roof(x, y, self.static_dz)
			else:
				z = -1*self.static_dz
			if not self.vertical:
				scale_x = 0.1*x
				scale_y = 0.1*y
				while(True):
					in_object = self._map.at_object_2d(x, y)
					if not in_object:
						break
					x -= scale_x
					y -= scale_y
			goal_position = np.array([x, y, z])
		if valid_point:
			self._x = goal_position[0]
			self._y = goal_position[1]
			self._z = goal_position[2]
		if not valid_point:
			self.reset(state)