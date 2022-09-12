# abstract class used to handle abstract components
from others.other import Other
from component import _init_wrapper
import random

# randomly set drone rotation
class Spawner(Other):

	@_init_wrapper
	def __init__(self, 
			  spawns_components,
			  drone_component='Drone',
			 ):
		self._rotating_index = 0

	def get_next_spawn(self):
		next_spawn = self._spawns[self._rotating_index]
		self._rotating_index += 1
		if self._rotating_index >= len(self._spawns):
			self._rotating_index = 0
		return next_spawn

	def spawn(self):
		next_spawn = self.get_next_spawn()
		position, yaw = next_spawn.get_spawn()
		self._drone.take_off()
		self._drone.teleport(position[0], position[1], position[2], yaw)

	def reset(self):
		self.spawn()

	# when using the debug controller
	def debug(self):
		self.spawn()