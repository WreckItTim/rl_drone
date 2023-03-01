from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import rl_utils as utils
import math

# gets orientation from a miscellaneous component
# yaw is expressed betwen [0, 2pi)  
# pass in a second component to get the orienation difference between the two
class Orientation(Sensor):
	
	@_init_wrapper
	def __init__(self,
				 misc_component,
				 misc2_component=None,
				 prefix = '',
				 transformers_components = None,
				 offline = False,
			  ):
		super().__init__(offline)

	def create_obj(self, data):
		observation = Vector(
			_data = data,
		)
		return observation

	# get state information from drone
	def step(self, state=None):
		data = []
		if self._misc2 is None:
			yaw = self._misc.get_yaw()
			if yaw < 0:
				yaw += 2*math.pi
			data.append(yaw)
		else:
			position1 = np.array(self._misc.get_position())
			position2 = np.array(self._misc2.get_position())
			distance_vector = position2 - position1
			yaw_1_2 = math.atan2(distance_vector[1], distance_vector[0])
			yaw1 = self._misc.get_yaw()
			yaw_diff = yaw_1_2 - yaw1
			if yaw_diff < 0:
				yaw_diff += 2*math.pi
			#yaw_diff = (yaw_diff + math.pi) % (2*math.pi) - math.pi
			data.append(yaw_diff)
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed