# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import math
import utils

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
			  ):
		pass

	# move at fixed rate
	def step(self, state=None):
		# must orient with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
		adjusted_x_speed = float(self.x_speed * math.cos(yaw) + self.y_speed * math.sin(yaw))
		adjusted_y_speed = float(self.y_speed * math.cos(yaw) + self.x_speed * math.sin(yaw))
		adjusted_z_speed = float(self.z_speed)
		# take movement
		self._drone.move(adjusted_x_speed, adjusted_y_speed, adjusted_z_speed, self.duration)