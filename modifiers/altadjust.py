from modifiers.modifier import Modifier
from component import _init_wrapper
import rl_utils as utils
import random

# this will adjust the altitude of a drone
# each step will insure alt is in a given range
# this is to account for the general tendency to fly up when going forward
# when the drone is only gien actions on the horizontal plane, this is needed
class AltAdjust(Modifier):

	@_init_wrapper
	def __init__(self, 
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  drone_component, # drone to spawn
			  order, # modify 'pre' or 'post'?
			  alt_min = -5, # range of allowable altitude
			  adjust = -4, # will teleport to this altitude if out of range
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0 , # keepts track of number of calls to parent method
			  ):
		super().__init__(base_component, parent_method, order, frequency, counter)

	# check and adjust altitude
	def activate(self, state):
		if self.check_counter(state):
			while(True):
				position = self._drone.get_position()
				if position[2] < self.alt_min:
					self._drone.teleport(position[0], position[1], self.adjust, self._drone.get_yaw())
				else:
					break
