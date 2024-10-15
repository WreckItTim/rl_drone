from spawners.spawner import Spawner
from component import _init_wrapper
import random
import numpy as np
import math
import rl_utils as utils
import pickle
import copy

# reads static list of paths, in a levels hiearchy (increasing difficulty)
# data structure of paths file (saved from pickle binary):
# {
# 	paths : [], # list of np.arrays where each np.array is a path of x,y,z points going from start to goal
# 	linearitys: [], # list of linear distances from start to goal for each path above
# 	nonlinearitys: [], # list of nonlinear values for each path = total_distance_traveled/linearity
# 	levels: {}, # dictionary, where each key is the level name (I just use the level numner as name) 
# 					# and where each value is a list of sublevels,
# 					# each sublevel above has a list of path indexes that belong to that sublevel
# }
class Levels(Spawner):
	# levels path is file path to data structure as outlined above
	# constructor
	@_init_wrapper
	def __init__(self,
				drone_component,
				levels_path,
				random_path = False,
				yaw_type = 0, # 'face': faces goal, 'random': random full range, value: specific yaw
				rotating_idx = 0,
				paths_per_sublevel=1,
				level = 1,
				start_level = 1,
				max_level = 1,
			):
		super().__init__()
		self._levels = utils.pk_read(levels_path)
		self.set_level(level)
		if not random_path:
			self._static_path_idxs = []
			for level in range(start_level, max_level+1):
				for sublevel in self._levels['levels'][level]:
					for i in range(paths_per_sublevel):
						self._static_path_idxs.append(sublevel[i])
			working_directory = utils.get_local_parameter('working_directory')
			pickle.dump(self._static_path_idxs, open(working_directory + '_static_path_idxs.p', 'wb'))


	def connect(self):
		super().connect()

	def set_level(self, level):
		self.level = level
		self._paths = self._levels['paths']
		self._nonlinearitys = self._levels['nonlinearitys']
		self._linearitys = self._levels['linearitys']
		self._sublevels = self._levels['levels']
		self.rotating_idx = 0
		self.reset_subs()

	def reset_subs(self):
		self._avaiable_subs = [i for i in range(len(self._sublevels[self.level]))]
		
	def reset_learning(self):
		self.rotating_idx = 0

	def spawn(self):
		if self.random_path:
			sublevel_idx = random.choice(self._avaiable_subs)
			del self._avaiable_subs[self._avaiable_subs.index(sublevel_idx)]
			if len(self._avaiable_subs) == 0:
				self.reset_subs()
			path_idx = random.choice(self._sublevels[self.level][sublevel_idx])
		else:
			path_idx = self._static_path_idxs[self.rotating_idx]
			self.rotating_idx = self.rotating_idx+1 if self.rotating_idx+1 < len(self._static_path_idxs) else 0   
			
		path = self._paths[path_idx]
		nonlinearity = self._nonlinearitys[path_idx]
		linearity = self._linearitys[path_idx]
		self._start_x, self._start_y, self._start_z = path[0]['position']
		self._goal_x, self._goal_y, self._goal_z = path[-1]['position']
		
		if self.yaw_type in ['face']:
			distance_vector = np.array([self._goal_x, self._goal_y, self._goal_z]) - np.array([self._start_x, self._start_y, self._start_z])
			self._start_yaw = math.atan2(distance_vector[1], distance_vector[0])
		elif self.yaw_type in ['random']:
			self._start_yaw = np.random.uniform(-1*math.pi, math.pi)
		else:
			self._start_yaw = self.yaw_type
			
		self._drone.teleport(self._start_x, self._start_y, self._start_z, self._start_yaw, ignore_collision=True, stabelize=True)
		
		return linearity, nonlinearity
		
	def reset(self, state=None):
		linearity, nonlinearity = self.spawn()
		if state is not None:
			state['path_linearity'] = linearity
			state['path_nonlinearity'] = nonlinearity
			state['goal_position'] = self.get_goal()
