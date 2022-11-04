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
				 xyz_point = [10, 0, 0],
				 random_point_on_train = False,
				 random_point_on_evaluate = False,
				 min_amp_up = 0, # increases min every reset by this much
				 max_amp_up = 0, # increases max every reset by this much
				 random_dim_min = 4, # magnitude of dim min
				 random_dim_max = 8, # magnitude of dim max
				 x_bounds = [-100, 100],
				 y_bounds = [-100, 100],
				 z_bounds = [0, 0],
				 random_yaw_on_train = False,
				 random_yaw_on_evaluate = False,
				 random_yaw_min = -1 * math.pi,
				 random_yaw_max = math.pi,
				 reset_on_step = False,
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
	
	def calculate_xyz(self, drone_position, relative_position, yaw, alpha):
		x = drone_position[0] + alpha * relative_position[0] * math.cos(yaw) + alpha * relative_position[1] * math.sin(yaw)
		y = drone_position[1] + alpha * relative_position[1] * math.cos(yaw) + alpha * relative_position[0] * math.sin(yaw)
		z = drone_position[2] + relative_position[2]
		in_object = self._map.at_object_2d(x, y)
		return x, y, z, in_object

	# need to recalculate relative point at each reset
	def reset(self, is_evaluation=False):
		drone_position = self._drone.get_position()
		# random point?
		relative_position = self.xyz_point.copy()
		random_point = False
		if is_evaluation and self.random_point_on_evaluate:
			random_point = True
		elif not is_evaluation and self.random_point_on_train:
			random_point = True
		if random_point:
			neg_pos = random.choice([-1, 1])
			x = neg_pos * random.uniform(self.random_dim_min, self.random_dim_max)
			relative_position[0] = min(self.x_bounds[1], max(self.x_bounds[0], x))
			neg_pos = random.choice([-1, 1])
			y = neg_pos * random.uniform(self.random_dim_min, self.random_dim_max)
			relative_position[1] = min(self.y_bounds[1], max(self.y_bounds[0], y))
			neg_pos = random.choice([-1, 1])
			z = neg_pos * random.uniform(self.random_dim_min, self.random_dim_max)
			relative_position[2] = min(self.z_bounds[1], max(self.z_bounds[0], z))
		# amp up max if training reset
		if not is_evaluation:
			self.random_dim_min += self.min_amp_up
			self.random_dim_max += self.max_amp_up
		# random yaw? # yaw counterclockwise rotation about z-axis
		if not is_evaluation and self.random_yaw_on_train:
			relative_yaw = random.uniform(self.random_yaw_min, self.random_yaw_max)
		elif is_evaluation and self.random_yaw_on_evaluate:
			relative_yaw = random.uniform(self.random_yaw_min, self.random_yaw_max)
		else:
			relative_yaw = self._drone.get_yaw()
		# shorten the distance until not in object
		alpha = 1
		in_object = True
		while in_object:
			if alpha < 0.1:
				self.reset(is_evaluation)
				break
			self._x, self._y, self._z, in_object = self.calculate_xyz(drone_position, relative_position, relative_yaw, alpha)
			alpha -= 0.1

	# if reset on each step
	def step(self, state):
		if self.reset_on_step:
			self.reset()
		state['goal_position'] = self.get_position()
		state['goal_yaw'] = self.get_yaw()

	# when using the debug controller
	def debug(self):
		self.reset()