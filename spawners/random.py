from spawners.spawner import Spawner
from component import _init_wrapper
import rl_utils as utils
import numpy as np
import math

# this will randomly spawn at a valid location
# the goal will also be randomly set to a valid location within a given relative range of the start location
class Random(Spawner):

	@_init_wrapper
	def __init__(self,
			drone_component, # drone to spawn
			roof_component, # used to detect highest collidable surface, get_roof(x,y) needs to be defined
			bounds_component, # valid range to spawn in 
			goal_range, # range of distances [meter] to create goal in range [min, max]
			vertical = False, # if allowed to randomize z spawn, otherwise will randomize x,y
			discretize = False, # spawns at integer values only
			yaw_type = 0, # 'face': faces goal, 'random': random full range, value: specific yaw
			dz = -4, # z-axis [meter] above ground (or rooftop of object if vertical) to spawn at
			):
		super().__init__()

	# select spawn then spawn drone
	def reset(self, state=None):
		self.random_start()
		self.random_goal(state)
		if self.yaw_type in ['face']:
			self.face_goal()
		elif self.yaw_type in ['random']:
			self.random_yaw()
		else:
			self.set_yaw(self.yaw_type)
		start_x, start_y, start_z = self.get_start()
		start_yaw = self.get_yaw()
		self._drone.teleport(start_x, start_y, start_z, start_yaw, ignore_collision=True, stabelize=True)
	
	def random_start(self):
		self._start_x, self._start_y, self._start_z = self.random_location()

	# set yaw to static value
	def set_yaw(self, yaw_val):
		self._start_yaw = yaw_val

	# uniform distribution of yaws
	def random_yaw(self):
		self._start_yaw = np.random.uniform(-1*np.pi, np.pi)

	# uniform distribution of yaws
	def face_goal(self):
		drone_pos = np.array(self.get_start())
		goal_pos = np.array(self.get_goal())
		displacement = goal_pos - drone_pos
		self._start_yaw = math.atan2(displacement[1], displacement[0])

	# uniform distribution between passed in range for x,y,z
	# considers if randomize z as well or not
	def random_location(self):
		if self.vertical:
			x, y, z = self._bounds.get_random()
			if self.discretize:
				x, y, z = int(x), int(y), int(z)
			# spawn above object
			z = self._roof.get_roof(x, y) + self.dz
		else:
			while (True):
				x, y, z = self._bounds.get_random()
				z = self.dz
				if self.discretize:
					x, y, z = int(x), int(y), int(z)
				# check if spawned in object
				in_object = self._roof.in_object(x, y, z)
				if not in_object:
					break
		return x, y, z

	def random_goal(self, state=None):
		start_x, start_y, start_z = self.get_start()
		drone_position = [start_x, start_y, start_z]
		# randomize until valid point
		attempt = 0
		while(True):
			r = np.random.uniform(self.goal_range[0], self.goal_range[1])
			theta = np.random.uniform(-1*np.pi, np.pi)
			delta_x = r * np.cos(theta)
			delta_y = r * np.sin(theta)
			goal_x = drone_position[0] + delta_x
			goal_y = drone_position[1] + delta_y
			# add vertical position to be on top of highest collidable object
			if self.vertical:
				goal_z = self._roof.get_roof(goal_x, goal_y) + self.dz
			else:
				goal_z = self.dz
			if self.discretize:
				goal_x, goal_y, goal_z = int(goal_x), int(goal_y), int(goal_z)
			# need to check if in object for non-vertical motion
			in_object = False
			if not self.vertical:
				# check if spawned in object
				in_object = self._roof.in_object(goal_x, goal_y, goal_z)
			# check if valid goal position
			if not in_object and self._bounds.check_bounds(goal_x, goal_y, goal_z):
				valid_point = True
				break
			# otherwise recheck (check for number of attempts and throw warning if exceeded)
			attempt += 1
			if attempt > 1000:
				utils.speak(f'could not find a valid goal at drone:{drone_position} and range:{self.goal_range} respawning...')
				valid_point = False
				break
		if valid_point:
			self._goal_x = goal_x
			self._goal_y = goal_y
			self._goal_z = goal_z
		else: # reset state to find new start spawn
			self.reset(state)