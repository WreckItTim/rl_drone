from actions.action import Action
from component import _init_wrapper

# rotates at given rate (radians/second) for given duration (seconds)
class FixedRotate(Action):
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  yaw_rate=0, 
			  duration=0,
			  ):
		pass

	# rotate yaw at fixed rate
	def step(self, state=None):
		#self._drone.rotate(self.yaw_rate, self.duration)
		current_position = self._drone.get_position() # meters
		current_yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
		target_yaw = current_yaw + self.yaw_rate
		self._drone.teleport(current_position[0], current_position[1], current_position[2], target_yaw, ignore_collision=False)