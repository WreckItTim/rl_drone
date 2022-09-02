# abstract class used to handle abstract components
from others.other import Other
from component import _init_wrapper
import random

# randomly set drone rotation
class Spawner(Other):

	@_init_wrapper
	def __init__(self, 
			  spawn_component,
			  drone_component='Drone', 
			  environment_component='Environment',
			  spawn_on_train=False,
			  spawn_on_evaluate=True,
			 ):
		pass

	def spawn(self):
		point, yaw = self._spawn.get_spawn()
		self._drone.teleport(point)
		self._drone.set_yaw(yaw)
		self._drone.take_off() 

	def reset(self):
		if not self._environment._evaluating and self.spawn_on_train:
			self.spawn()
		if self._environment._evaluating and self.spawn_on_evaluate:
			self.spawn()

	# when using the debug controller
	def debug(self):
		self.spawn()