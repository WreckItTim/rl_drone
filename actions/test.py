# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import numpy as np
import math
import utils

class Test(Action):
	# rotates at given rate (radians/second) for given duration (seconds)
	@_init_wrapper
	def __init__(self, 
			  drone_component, 
			  goal_component, 
			  ):
		super().__init__()

	def act(self):
		goal_position = np.array(self._goal.get_position())
		drone_position = np.array(self._drone.get_position())
		distance_vector = goal_position - drone_position
		yaw_to_goal = utils.position_to_yaw(distance_vector)
		self._drone.teleport(drone_position[0], drone_position[1], drone_position[2], yaw_to_goal)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')