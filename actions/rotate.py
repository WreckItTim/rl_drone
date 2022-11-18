# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import numpy as np
import math

class Rotate(Action):
	# rotates at given rate (radians/second) for given duration (seconds)
	# this is a continuous action that will scale the input yaw_rate by the rl_output
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  base_yaw_rate, 
			  zero_min_threshold=-0.1,
			  zero_max_threshold=0.1,
			  duration=2
			  ):
		super().__init__()
		self._min_val = -1
		self._max_val = 1

	def act(self, rl_output):
		# get speed magnitude from rl_output
		if rl_output > self.zero_min_threshold and rl_output < self.zero_max_threshold:
			return
		self._drone.rotate(rl_output*self.base_yaw_rate, self.duration)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')
