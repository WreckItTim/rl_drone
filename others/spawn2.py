from others.other import Other
from component import _init_wrapper
import pickle
import random

# data structure specifying a spawn zone
class Spawn2(Other):
	# pass in either static x, y, z, yaw
	# or ranges for random values
	# constructor
	@_init_wrapper
	def __init__(self, 
				read_path, # read in dict of possible paths or static spawns
				random, # True will get random path, False will use static
				nSteps=1, # if random (how many steps to sample goal)
				max_steps=20,
				clip_spawns=-1,
				 ):
		super().__init__()

	def connect(self):
		super().connect()
		if self.random:
			self._dicts = pickle.load(open(self.read_path, 'rb'))
			self._idxs = {}
			# sort
			for i, d in enumerate(self._dicts):
				steps = min(self.max_steps, len(d['a_path'])-1)
				for s in range(steps, 0, -1):
					if s not in self._idxs:
						self._idxs[s] = []
					self._idxs[s].append(i)
			self._last_state = self.get_random()
		else:
			self._spawns = pickle.load(open(self.read_path, 'rb'))[:self.clip_spawns]
			self._idx = 0
			self._last_state = self.get_static()
		self._redo = False

	# uniform distribution between passed in range
	def get_random_pos(self):
		if self.vertical:
			x, y, z = self._bounds.get_random()
			z = self._map.get_roof(x, y, self.dz)
		else:
			while (True):
				x, y, z = self._bounds.get_random()
				z = -1*self.dz
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