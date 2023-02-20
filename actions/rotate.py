from actions.action import Action
from component import _init_wrapper

# rotates at continuous rate (radians/second) for given duration (seconds)
class Rotate(Action):
	# this is a continuous action that will scale the input yaw_rate by the rl_output
	@_init_wrapper
	def __init__(self, 
				drone_component, 
				base_yaw_deg, # degrees to rotate in range [-180, 180]
				zero_threshold=0, # absolute value below this will do nothing (true zero)
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = -1,
				max_space = 1,
			):
				self.min_space = min_space
				self.max_space = max_space
		
	# rotate yaw at input rate
	def step(self, state):
		rl_output = state['rl_output'][self._idx]
		# check for true zero
		if abs(rl_output) < self.zero_threshold:
			return 'rotate(true_zero)'
		# rotate calculated rate from rl_output
		adjusted_yaw_deg = rl_output*self.base_yaw_deg
		self._drone.rotate(adjusted_yaw_deg)
		return f'rotate({adjusted_yaw_deg})'