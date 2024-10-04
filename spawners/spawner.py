from component import Component
import rl_utils as utils


class Spawner(Component):
	# constructor
	def __init__(self):
		pass

	# set start and target location on reset()
	def reset(self, state=None):
		raise NotImplementedError

	# get x,y,z,yaw of last spawned starting location
	def get_start(self):
		return [self._start_x, self._start_y, self._start_z]

	# get x,y,z of last spawned target location
	def get_goal(self):
		return [self._goal_x, self._goal_y, self._goal_z]

	# this is used some times in arbitrary components
	def get_position(self):
		return self.get_goal()

	def get_yaw(self):
		return self._start_yaw

	def connect(self):
		super().connect()