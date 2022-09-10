from turtle import position
from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import math
import utils

# gets information relative to objective point
class Goal(Sensor):

	# specify which state values to get
	@_init_wrapper
	def __init__(self,
                 drone_component, 
                 goal_component, 
				 get_yaw_difference = True,
				 transformers_components = None,
				 offline = False,
			  ):
		super().__init__(offline)
		
	# get information reltaive between current and objective point
	def sense(self):
		data = []
		names = []
		if self.get_yaw_difference:
			# get drone yaw
			drone_yaw = self._drone.get_yaw()
			objective_yaw = utils.position_to_yaw(self._goal.xyz_point)
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