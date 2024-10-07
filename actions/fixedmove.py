# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import math
import rl_utils as utils
import numpy as np

# translates forward at given distance
class FixedMove(Action):
	# constructor takes 4d array where first 3 components are distance, 4th is at what speed
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  x_distance = 0, 
			  y_distance = 0, 
			  z_distance = 0, 
			  speed = 2, # only used if not teleporting
			  adjust_for_yaw = False,
			  ):
		pass

	# move at fixed rate
	def step(self, state=None, execute=True):
		# must orient with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
		if self.adjust_for_yaw:
			adjusted_x_distance = float(self.x_distance * math.cos(yaw) - self.y_distance * math.sin(yaw))
			adjusted_y_distance = float(self.x_distance * math.sin(yaw) + self.y_distance * math.cos(yaw))
		else:
			adjusted_x_distance= float(self.x_distance)
			adjusted_y_distance = float(self.y_distance)
		adjusted_z_distance = float(self.z_distance)
		# take movement
		if execute:
			self._drone.move(adjusted_x_distance, adjusted_y_distance, adjusted_z_distance, self.speed)
		return {'x':adjusted_x_distance, 'y':adjusted_y_distance, 'z':adjusted_z_distance}
		