from actions.action import Action
from component import _init_wrapper
import math

# continuous move forward that will scale the input x,y,z speeds by the rl_output
class Move(Action):
	# constructor takes 4d array where first 3 components are direction and 4th component is speed
	# note that speed is an arbitrary unit that is defined by the drone
	# zero threshold determines when to not move forward - sets a real 0 value
	@_init_wrapper
	def __init__(self, 
			  drone_component,
			  base_x_speed=0, 
			  base_y_speed=0, 
			  base_z_speed=0, 
			  zero_threshold=0, # below this will do nothing (true zero)
			  duration=2,
			  ):
		# set these values for continuous actions
		# they determine the possible ranges of output from rl algorithm
		self._min_space = -1
		self._max_space = 1
		
	# move at input rate
	def step(self, state):
		rl_output = state['rl_output'][self._idx]
		# check for true zero
		if abs(rl_output) < self.zero_threshold:
			return 'move(true_zero)'
		# must orient self with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
		# calculate rate from rl_output
		adjusted_x_speed = float(rl_output * (self.base_x_speed * math.cos(yaw) - self.base_y_speed * math.sin(yaw)))
		adjusted_y_speed = float(rl_output * (self.base_x_speed * math.sin(yaw) + self.base_y_speed * math.cos(yaw)))
		adjusted_z_speed = float(rl_output * self.base_z_speed)
		# move calculated rate
		has_collided = self._drone.move(adjusted_x_speed, adjusted_y_speed, adjusted_z_speed, self.duration)
		state['has_collided'] = has_collided
		return f'move({adjusted_x_speed}, {adjusted_y_speed}, {adjusted_z_speed})'