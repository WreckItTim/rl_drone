# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import math
import rl_utils as utils
import numpy as np

# translates forward at given rate (meters/second) for given duration (seconds)
class FixedMove(Action):
	# constructor takes 4d array where first 3 components are direction and 4th component is speed
	# note that speed is an arbitrary unit that is defined by the drone
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  x_speed=0, 
			  y_speed=0, 
			  z_speed=0, 
			  duration=0,
			  adjust_for_yaw = False,
			  ):
		pass

	# move at fixed rate
	def step(self, state=None):
		# must orient with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
		if self.adjust_for_yaw:
			adjusted_x_speed = float(self.x_speed * math.cos(yaw) - self.y_speed * math.sin(yaw))
			adjusted_y_speed = float(self.x_speed * math.sin(yaw) + self.y_speed * math.cos(yaw))
		else:
			adjusted_x_speed = float(self.x_speed)
			adjusted_y_speed = float(self.y_speed)
		adjusted_z_speed = float(self.z_speed)
		# take movement
		#has_collided = self._drone.move(adjusted_x_speed, adjusted_y_speed, adjusted_z_speed, self.duration)
		current_position = self._drone.get_position() # meters
		target_position = current_position + np.array([adjusted_x_speed, adjusted_y_speed, adjusted_z_speed], dtype=float)
		self._drone.teleport(target_position[0], target_position[1], target_position[2], yaw, ignore_collision=False)