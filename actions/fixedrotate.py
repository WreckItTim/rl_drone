# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import numpy as np
import math

class FixedRotate(Action):
	# rotates at given rate (radians/second) for given duration (seconds)
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  yaw_rate, 
			  duration=2
			  ):
		super().__init__()

	def act(self):
		self._drone.rotate(self.yaw_rate, self.duration)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')