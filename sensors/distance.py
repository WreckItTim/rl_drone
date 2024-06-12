from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import math

# gets position from a miscellaneous component
# can send in a second component to get positoin and distance between two
class Distance(Sensor):
	
	@_init_wrapper
	def __init__(self,
                 misc_component, 
                 misc2_component, 
				 include_x = True,
				 include_y = True,
				 include_z = True,
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
		
	# get information reltaive between current and objective point
	def step(self, state=None):
		data = []
		position1 = np.array(self._misc.get_position())
		position2 = np.array(self._misc2.get_position())
		distance_vector = position2 - position1
		distance = 0
		if self.include_x:
			distance += distance_vector[0]**2
		if self.include_y:
			distance += distance_vector[1]**2
		if self.include_z:
			distance += distance_vector[2]**2
		data.append(math.sqrt(distance))

		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed