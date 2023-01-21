from actions.action import Action
from component import _init_wrapper

# rotates at given rate (radians/second) for given duration (seconds)
class FixedRotate(Action):
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  yaw_rate, 
			  duration=2
			  ):
		pass

	# rotate yaw at fixed rate
	def step(self, state=None):
		self._drone.rotate(self.yaw_rate, self.duration)