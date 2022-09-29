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
			  zero_threshold=0.25,
			  duration=2
			  ):
		super().__init__()

	def act(self, rl_output):
		# get speed magnitude from rl_output
		magnitude = rl_output
		if rl_output > -1*self.zero_threshold and rl_output < self.zero_threshold:
			magnitude = 0
		self._drone.rotate(magnitude*self.base_yaw_rate, self.duration)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')