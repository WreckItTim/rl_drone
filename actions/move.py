# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import numpy as np
import math

class Move(Action):
	# constructor takes 4d array where first 3 components are direction and 4th component is speed
	# note that speed is an arbitrary unit that is defined by the drone
	# this is a continuous action that will scale the input x,y,z speeds by the rl_output
	@_init_wrapper
	def __init__(self, 
			  drone_component,
			  base_x_speed=0, 
			  base_y_speed=0, 
			  base_z_speed=0, 
			  zero_threshold=0.25,
			  duration=2
			  ):
		super().__init__()

	def act(self, rl_output):
		# get speed magnitude from rl_output
		magnitude = rl_output
		if rl_output < self.zero_threshold:
			return
		# must orient self with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotationa bout z-axis
		adjusted_x_speed = float(magnitude * self.base_x_speed * math.cos(yaw) + magnitude * self.base_y_speed * math.sin(yaw))
		adjusted_y_speed = float(magnitude * self.base_y_speed * math.cos(yaw) + magnitude * self.base_x_speed * math.sin(yaw))
		adjusted_z_speed = float(magnitude * self.base_z_speed)
		self._drone.move(adjusted_x_speed, adjusted_y_speed, adjusted_z_speed, self.duration)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')
