# discrete move in one direction
from actions.action import Action
from component import _init_wrapper
import numpy as np
import math

class FixedMove(Action):
	# constructor takes 4d array where first 3 components are direction and 4th component is speed
	# note that speed is an arbitrary unit that is defined by the drone
	@_init_wrapper
	def __init__(self, drone_component, x_speed, y_speed, z_speed, duration):
		super().__init__()
		self._x_speed = x_speed
		self._y_speed = y_speed
		self._z_speed = z_speed

	def act(self):
		self._drone.move(self.x_speed, self.y_speed, self.z_speed, self.duration)

	# uses a string to fetch a preset movement (left, right, ...)
	# movements are addititve with underscores in move_component (to make diagnols)
	@staticmethod
	def get_move(drone_component, move_type, step_size, duration):
		x_speed, y_speed, z_speed = 0, 0, 0
		moves = move_type.lower().split('_')
		for move in moves:
			if 'left' in move: y_speed -= step_size
			if 'right' in move: y_speed += step_size
			if 'up' in move: z_speed -= step_size
			if 'down' in move: z_speed += step_size
			if 'forward' in move: x_speed += step_size
			if 'backward' in move: x_speed -= step_size
		return FixedMove(
			drone_component=drone_component, 
			x_speed=x_speed, 
			y_speed=y_speed, 
			z_speed=z_speed,
			duration=duration, 
			name=move_type,
		)
	
	def reset(self):
		# must orient self with yaw
		yaw = self._drone.get_yaw() # yaw counterclockwise rotationa bout z-axis
		self.x_speed = float(self._x_speed * math.cos(yaw) + self._y_speed * math.sin(yaw))
		self.y_speed = float(self._y_speed * math.cos(yaw) + self._x_speed * math.sin(yaw))
		self.z_speed = float(self._z_speed)
		
	# when using the debug controller
	def debug(self):
		self.act()
		return(f'position {self._drone.get_position()} yaw {self._drone.get_yaw()}')