from actions.action import Action
from component import _init_wrapper

# rotates at continuous rate (radians/second) for given duration (seconds)
class Rotate(Action):
	# this is a continuous action that will scale the input yaw_rate by the rl_output
	@_init_wrapper
	def __init__(self, 
				drone_component, 
				base_yaw_rate, 
				zero_threshold=0, # below this will do nothing (true zero)
				duration=2,
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = 0,
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
		adjusted_rate = rl_output*self.base_yaw_rate
		self._drone.rotate(adjusted_rate, self.duration)
		return f'rotate({adjusted_rate})'