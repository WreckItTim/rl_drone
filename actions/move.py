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
				speed=2, # m/s
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = -1, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				adjust_for_yaw = False,
				active = True, # False will just return default behavior
				abs_zero = 0.02, # give some room for error on predicting zero
				zero_thresh_abs=True,
			):
		pass
		
	# move at input rate
	def step(self, state, execute=True):
		if not self.active:
			return {}
		rl_output = state['rl_output'][self._idx]
		# check for true zero
		if self.zero_thresh_abs:
			if abs(rl_output) < self.abs_zero:
				return {}
		else:
			if rl_output <= self.abs_zero:
				return {}
		x_rel = rl_output * self.base_x_rel
		y_rel = rl_output * self.base_x_rel
		z_rel = rl_output * self.base_x_rel
		# calculate rate from rl_output
		if self.adjust_for_yaw:
			# must orient self with yaw
			yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
			adjusted_x_rel = float((x_rel * math.cos(yaw) - y_rel * math.sin(yaw)))
			adjusted_y_rel = float((x_rel * math.sin(yaw) + y_rel * math.cos(yaw)))
		else:
			adjusted_x_rel = float(x_rel)
			adjusted_y_rel = float(y_rel)
		adjusted_z_rel = float(z_rel)
		# move calculated rate
		if execute:
			self._drone.move(adjusted_x_rel, adjusted_y_rel, adjusted_z_rel, self.speed)
		return {'x':adjusted_x_rel, 'y':adjusted_y_rel, 'z':adjusted_z_rel}