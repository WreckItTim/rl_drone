from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import utils
import math

# gets orientation from a miscellaneous component
# yaw is expressed betwen [0, 2pi)  
# pass in a second component to get the orienation between the two
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

	# get state information from drone
	def sense(self):
		data = []
		names = []
		if self._misc2 is None:
			yaw = self._misc.get_yaw()
			data.append(yaw)
			names.append(self.prefix+'_yaw')
		else:
			position1 = np.array(self._misc.get_position())
			position2 = np.array(self._misc2.get_position())
			distance_vector = position2 - position1
			yaw_1_2 = utils.position_to_yaw(distance_vector)
			yaw1 = self._misc.get_yaw()
			yaw_diff = math.pi - abs(abs(yaw_1_2 - yaw1) - math.pi)
			print('yaw_diff', yaw_diff)
			data.append(yaw_diff)
			names.append(self.prefix+'_yaw_diff')
		observation = Vector(
			_data = data,
			names = names,
		)
		return self.transform(observation)