from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math
import utils

# TODO: implement
# applies a penalty for moving closer to previously traveled point(s)
class Goal(Reward):
	# constructor, set the relative point and min-max distances to normalize by
	@_init_wrapper
	def __init__(self,
				 drone_component,
				 max_distance, # distance scaling factor (max farthest away from ALL points)
				 nPoints, # number of previous points to calc distance from
				 include_z=True, # include z-axis in calculations
				 terminate=True, # uf True will terminate episodes when Goal
				 ):
		super().__init__()
		self._nDims = 3
		if not self.include_z:
			self._nDims = 2

	def get_distance(self):
		_drone_position = np.array(self._drone.get_position(), dtype=float)
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
		distance_matrix = self._points - _drone_position
		distance = np.linalg.norm(distance_matrix)
		return distance
	
	# get reward based on distance to point 
	def step(self, state):
		# TODO
		return None
		#return value, done and self.terminate

	def reset(self, state):
		# fill all points with starting location
		self._start_position = self._drone.get_position() # x,y,z list
		self._points = np.zeros((self._nDims, self.nPoints), dtype=float)
		self._points[:] = self._start_position
		self._rotating_idx = 0