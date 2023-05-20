from others.other import Other
from component import _init_wrapper
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
				min_r = 0,
				mean_r = 0,
				std_r = 0,
				min_dz = 0,
				mean_dz = 0,
				std_dz = 0,
				vertical = True,
			):
		
		self._x = mean_r * np.cos(0)
		self._y = mean_r * np.sin(0)
		self._z = mean_dz

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
	def start(self, state):
		drone_position = self._drone.get_position()
		drone_yaw = self._drone.get_yaw()
		self.get_goal(drone_position, drone_yaw)

	def get_goal(self, drone_position, drone_yaw, 
		r=None, dz=None, yaw=None,
	):
		# checking voxels...
		z = -4
		print('VOXELS...')
		for x in range(-125, 125):
			for y in range(-125, 125):
				obj = self._map.at_object_2d(x, y)
				if obj:
					print(x, y)
		# randomize until get valid point
		attempt = 0
		while(True):
			if r is None:
				r = np.random.normal(self.mean_r, self.std_r)
				r = max(r, self.min_r)
			if dz is None:
				dz = np.random.normal(self.mean_dz, self.std_dz)
				dz = max(dz, self.min_dz)
			if yaw is None:
				yaw = np.random.uniform(-1*np.pi, np.pi)
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
			bound_check = self._bounds.check_bounds(x, y, z)
			if not in_object and bound_check:
				break
			attempt += 1
			if attempt > 1000:
				utils.speak(f'ERR could not find goal in bounds at {drone_position}')
				utils.speak(f'{goal_position}')
				utils.speak(f'{[r, dz, yaw, in_object, bound_check]}')
				x = input()
		self.set_position(
			goal_position[0],
			goal_position[1],
			goal_position[2],
		)