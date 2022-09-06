from turtle import position
from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import math

# gets information relative to objective point
class Objective(Sensor):

	# specify which state values to get
	@_init_wrapper
	def __init__(self,
				 drone_component,
				 xyz_point,
				 get_yaw_difference = True,
				 transformers_components=None,
				 ):
		super().__init__()
		self.xyz_point = np.array(xyz_point, dtype=float)
		self._x = self.xyz_point[0]
		self._y = self.xyz_point[1]
		self._z = self.xyz_point[2]
		
	# get information reltaive between current and objective point
	def sense(self):
		data = []
		names = []
		if self.get_yaw_difference:
			# get drone yaw
			drone_yaw = self._drone.get_yaw()
			# get magniute of angle for objective yaw relative origin
			if self.xyz_point[0] == 0:
				if self.xyz_point[1] == 0:
					objective_yaw = 0
				elif self.xyz_point[1] > 0:
					objective_yaw = math.pi/2
				else:
					objective_yaw = 3*math.pi/2
			elif self.xyz_point[1] == 0:
				if self.xyz_point[0] > 0:
					objective_yaw = 0
				else:
					objective_yaw = math.pi
			else:
				objective_yaw_magnitude = abs(math.atan(self.xyz_point[1] / self.xyz_point[0]))
				# get exact angle depending on quadrant
				if self.xyz_point[1] > 0:
					# yaw quad 1
					if self.xyz_point[0] > 0:
						objective_yaw = objective_yaw_magnitude
					# yaw quad 2
					if self.xyz_point[0] < 0:
						objective_yaw = objective_yaw_magnitude + math.pi/2
				else:
					# yaw quad 3
					if self.xyz_point[0] < 0:
						objective_yaw = objective_yaw_magnitude + math.pi
					# yaw quad 4
					if self.xyz_point[0] > 0:
						objective_yaw = objective_yaw_magnitude + 3*math.pi/2
			# get yaw differnece
			yaw_difference = objective_yaw - drone_yaw
			if yaw_difference < 0:
				yaw_difference += 2*math.pi
			data.append(yaw_difference)
			names.append('yaw_difference')
		observation = Vector(
			_data = data,
			names = names,
		)
		return self.transform(observation)

	def reset(self):
		position = self._drone.get_position()
		yaw = self._drone.get_yaw() 
		x = position[0] + self._x * math.cos(yaw) + self._y * math.sin(yaw)
		y = position[1] + self._y * math.cos(yaw) + self._x * math.sin(yaw)
		z = position[2] + self._z
		self.xyz_point = np.array([x, y, z], dtype=float)