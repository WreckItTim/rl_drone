from others.other import Other
from component import _init_wrapper
import pickle
import random

# datstruct to keep track of goal stats
class Goal(Other):

	@_init_wrapper
	def __init__(self, 
			):
		self._xyz = None
		self._steps = None

	def set_position(self, xyz):
		self._xyz = xyz.copy()

	def get_position(self):
		return self._xyz.copy()

	def set_steps(self, steps):
		self._steps = steps

	def get_steps(self):
		return self._steps