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
				base_x_rel=0, # relative x,y,z to drone
				base_y_rel=0, 
				base_z_rel=0, 
				zero_threshold=0, # absolute value of rl_output below this will do nothing (true zero)
				speed=2, # m/s
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = 0, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
			):
				self.min_space = min_space
				self.max_space = max_space
		
	# move at input rate
	def step(self, state):
		rl_output = state['rl_output'][self._idx]
		# check for true zero
		if abs(rl_output) < self.zero_threshold:
			return 'move(true_zero)'
		# must orient self with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
		# calculate rate from rl_output
		adjusted_x_rel = float(rl_output * (self.base_x_rel * math.cos(yaw) - self.base_y_rel * math.sin(yaw)))
		adjusted_y_rel = float(rl_output * (self.base_x_rel * math.sin(yaw) + self.base_y_rel * math.cos(yaw)))
		adjusted_z_rel = float(rl_output * self.base_z_rel)
		# move calculated rate
		self._drone.move(adjusted_x_rel, adjusted_y_rel, adjusted_z_rel, self.speed)
		return f'move({adjusted_x_rel}, {adjusted_y_rel}, {adjusted_z_rel})'