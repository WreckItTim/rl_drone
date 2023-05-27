from others.other import Other
from component import _init_wrapper
import pickle
import random

# set drone spawn and goal from list
class Spawn(Other):

	@_init_wrapper
	def __init__(self, 
				read_path, # read in dict of possible paths or static spawns
				random, # True will get random path, False will use static
				nSteps=1, # if random (how many steps to sample goal)
				max_steps=20,
			):
		pass

	def connect(self, state=None):
		super().connect(state)
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
		else:
			self._spawns = pickle.load(open(self.read_path, 'rb'))
			self._idx = 0

	# need to recalculate relative point at each reset
	def start(self, state=None):
		if self.random:
			idx = random.choice(self._idxs[self.nSteps])
			dic =  self._dicts[idx]
			path = dic['a_path']
			start = random.randint(0, len(path)-self.nSteps-1)
			if start < self.nSteps:
				end = start + self.nSteps
			elif start >= len(path) - self.nSteps:
				end = start - self.nSteps
			else:
				flip = random.choice([-1, 1])
				end = start + flip*self.nSteps
			drone_position = path[start][:3]
			yaw = path[start][3]
			goal_position = path[end][:3]
			astar_steps = self.nSteps
		else:
			drone_position = self._spawns[self._idx][0][:3]
			yaw = self._spawns[self._idx][0][3]
			goal_position = self._spawns[self._idx][1][:3]
			astar_steps = self._spawns[self._idx][2]
			self._idx += 1
			if self._idx >= len(self._spawns):
				self._idx = 0
		return drone_position.copy(), yaw, goal_position.copy(), astar_steps