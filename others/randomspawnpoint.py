# abstract class used to handle abstract components
from others.other import Other
from component import _init_wrapper
import random

# randomly spawns drone in one of the indicated safe zones at a random angle
class RandomSpawnPoint(Other):
	# zone: each safe zone is a rectangle min-max xyz zone, which the drone will randomly spawn at one point inside of it 
		# note you can shrink this to a point by setting min=max

	@_init_wrapper
	def __init__(self, drone_component, environment_component, spawn_zones_components):
		pass

	def reset(self):
		if not self._environment._evaluating:
			_spawn_zone = random.choice(self._spawn_zones)
			spawn_point = _spawn_zone.random_point()
			self._drone.teleport(spawn_point)

	# when using the debug controller
	def debug(self):
		self.reset()