from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np

# gets position from a miscellaneous component
# can send in a second component to get positoin and distance between two
class Distance(Sensor):
	
	@_init_wrapper
	def __init__(self,
                 misc_component, 
                 misc2_component, 
				 prefix = '',
				 transformers_components = None,
				 offline = False,
			  raw_code=None,
			  ):
		super().__init__(offline, raw_code)

	def create_obj(self, data):
		observation = Vector(
			_data = data,
		)
		return observation
		
	# get information reltaive between current and objective point
	def sense2(self):
		data = []
		position1 = np.array(self._misc.get_position())
		position2 = np.array(self._misc2.get_position())
		distance_vector = position2 - position1
		distance = np.linalg.norm(distance_vector)
		data.append(distance)

		observation = self.create_obj(data)
		return observation