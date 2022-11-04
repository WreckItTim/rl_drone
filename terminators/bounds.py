# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np
import math

class Bounds(Terminator):
	# constructor
	@_init_wrapper
	def __init__(self, 
				 drone_component, 
				 x_bounds,
				 y_bounds,
				 z_bounds,
	):
		super().__init__()

	# checks if within distance of point
	def terminate(self, state):
		_drone_position = self._drone.get_position()
		x = _drone_position[0]
		y = _drone_position[1]
		z = _drone_position[2]
		if x < self.x_bounds[0] or x > self.x_bounds[1]:
			state['termination_reason'] = 'bounds'
			state['termination_result'] = 'failure'
			return True
		if y < self.y_bounds[0] or y > self.y_bounds[1]:
			state['termination_reason'] = 'bounds'
			state['termination_result'] = 'failure'
			return True
		if z < self.z_bounds[0] or y > self.z_bounds[1]:
			state['termination_reason'] = 'bounds'
			state['termination_result'] = 'failure'
			return True
		return False