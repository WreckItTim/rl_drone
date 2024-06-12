from modifiers.modifier import Modifier
from component import _init_wrapper
import rl_utils as utils
import random

# this will select from several spawn objects and move drone there
# option is to randomly select from passed in spawn objects or static rotating queue
# spawn objects are defined in others
class Spawner(Modifier):

	@_init_wrapper
	def __init__(self, 
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  drone_component, # drone to spawn
			  spawns_components, # list of spawn objects
			  order, # modify 'pre' or 'post'?
			  random = False, # set true to select randomly from spawn objects
			  on_evaluate = True, # toggle to run modifier on evaluation environ
			  on_train = True, # toggle to run modifier on train environ
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = True, # will activate on first call otherwise only if % is not 0
			 ):
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)
		self._rotating_index = 0

	# if random, randomly select from list of spawn objects
	# if not random, rotate through queue from list
	def get_next_spawn(self):
		if self.random:
			random_index = random.randint(0, len(self._spawns) - 1)
			next_spawn = self._spawns[random_index]
		else:
			next_spawn = self._spawns[self._rotating_index]
			self._rotating_index += 1
			if self._rotating_index >= len(self._spawns):
				self._rotating_index = 0
		return next_spawn

	# select spawn then spawn drone
	def activate(self, state):
		if self.check_counter(state):
			next_spawn = self.get_next_spawn()
			position, yaw = next_spawn.get_spawn()
			#print('spawn to ', position[0], position[1], position[2], yaw)
			self._drone.teleport(position[0], position[1], position[2], yaw)
			#print('spawned to ', self._drone.get_position(), self._drone.get_yaw())
