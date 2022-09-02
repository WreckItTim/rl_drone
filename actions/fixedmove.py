# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import numpy as np
import math

class FixedMove(Action):
	# constructor takes 4d array where first 3 components are direction and 4th component is speed
	# note that speed is an arbitrary unit that is defined by the drone
	@_init_wrapper
	def __init__(self, drone_component, x_distance, y_distance, z_distance, speed):
		super().__init__()

	def act(self):
		# must orient self with yaw
		yaw = self._drone.get_yaw(radians=True) # yaw counterclockwise rotationa bout z-axis
		x = self.x_distance * math.cos(yaw) + self.y_distance * math.sin(yaw)
		y = self.y_distance * math.cos(yaw) + self.x_distance * math.sin(yaw)
		z = self.z_distance
		self._drone.move(np.array([x, y, z], dtype=float), self.speed)

	# uses a string to fetch a preset movement (left, right, ...)
	# movements are addititve with underscores in move_component (to make diagnols)
	@staticmethod
	def get_move(drone_component, move_type, step_size, speed):
		x_distance, y_distance, z_distance = 0, 0, 0
		moves = move_type.lower().split('_')
		for move in moves:
			if 'left' in move: y_distance -= step_size
			if 'right' in move: y_distance += step_size
			if 'up' in move: z_distance -= step_size
			if 'down' in move: z_distance += step_size
			if 'forward' in move: x_distance += step_size
			if 'backward' in move: x_distance -= step_size
		return FixedMove(
			drone_component=drone_component, 
			x_distance=x_distance, 
			y_distance=y_distance, 
			z_distance=z_distance,
			speed=speed, 
			name=move_type,
		)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')