from actions.action import Action
from component import _init_wrapper

# rotates at continuous rate (radians/second) for given duration (seconds)
class Rotate(Action):
	# this is a continuous action that will scale the input yaw_rate by the rl_output
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  base_yaw_rate, 
			  zero_min_threshold=0, # above this
			  zero_max_threshold=0, # and below this will do nothing (true zero)
			  duration=2,
			  ):
		self._min_space = -1
		self._max_space = 1
		
	# rotate yaw at input rate
	def step(self, state):
		rl_output = state['rl_output'][self._idx]
		# check for true zero
		if rl_output > self.zero_min_threshold and rl_output < self.zero_max_threshold:
			return 'rotate(true_zero)'
		# rotate calculated rate from rl_output
		adjusted_rate = rl_output*self.base_yaw_rate
		self._drone.rotate(adjusted_rate, self.duration)
		return f'rotate({adjusted_rate})'