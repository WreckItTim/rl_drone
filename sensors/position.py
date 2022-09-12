from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper

# gets position from a miscellaneous component
class Position(Sensor):
	
	@_init_wrapper
	def __init__(self,
                 misc_component, 
				 prefix = '',
				 transformers_components = None,
				 offline = False,
			  ):
		super().__init__(offline)
		
	# get information reltaive between current and objective point
	def sense(self):
		data = []
		names = []
		position = self._misc.get_position()
		data.append(position[0])
		names.append(self.prefix+'_x')
		data.append(position[1])
		names.append(self.prefix+'_y')
		data.append(position[2])
		names.append(self.prefix+'_z')
		observation = Vector(
			_data = data,
			names = names,
		)
		return self.transform(observation)