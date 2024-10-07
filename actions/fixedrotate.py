from actions.action import Action
from component import _init_wrapper

# rotates a given yaw difference
class FixedRotate(Action):
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  yaw_diff = 0,
			  ):
		pass

	# rotate yaw at fixed rate
	def step(self, state=None, execute=True):
		if execute:
			self._drone.rotate(self.yaw_diff)
		return {'yaw':self.yaw_diff}