from actions.action import Action
from component import _init_wrapper

# rotates at continuous rate (radians/second) for given duration (seconds)
class Rotate(Action):
	# this is a continuous action that will scale the input yaw_rate by the rl_output
	@_init_wrapper
	def __init__(self, 
				drone_component, 
				base_yaw, # radians to rotate in range [-pi, pi] (angle to sweep, relative to drone)
				zero_threshold=0.01,# absolute value below this will do nothing (true zero)
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = -1,
				max_space = 1,
			):
		pass
		
	# rotate yaw at input rate
	def step(self, state, execute=True):
		rl_output = state['rl_output'][self._idx]
		# check for true zero
		if abs(rl_output) < self.zero_threshold:
			return {}
		# rotate calculated rate from rl_output
		adjusted_yaw = rl_output*self.base_yaw
		if execute:
			self._drone.rotate(adjusted_yaw)
		return {'yaw':adjusted_yaw}