# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np
import math
import numpy as np

class Distance(Terminator):
	# constructor
	@_init_wrapper
	def __init__(self, 
                 drone_component,
				 goal_component,
				 include_z=False,
				 max_distance=100,
				 ):
		super().__init__()

	def get_distance(self):
		_drone_position = np.array(self._drone.get_position(), dtype=float)
		_goal_position = np.array(self._goal.get_position(), dtype=float)
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
			_goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
		distance_vector = _goal_position - _drone_position
		distance = np.linalg.norm(distance_vector)
		return distance

	# checks if within distance of point
	def terminate(self, state):
		distance = self.get_distance()

		if distance > self.max_distance:
			state['termination_reason'] = 'max_distance'
			state['termination_result'] = 'failure'
			return True
		return False