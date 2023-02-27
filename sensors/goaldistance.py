from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import math

# gets position from a miscellaneous component
# can send in a second component to get positoin and distance between two
class GoalDistance(Sensor):
	
	@_init_wrapper
	def __init__(self,
                drone_component, 
                goal_component, 
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
		_drone_position = np.array(self._drone.get_position())
		distance_vector = self._goal_position - _drone_position
		distance = 0
		if self.include_x:
			distance += distance_vector[0]**2
		if self.include_y:
			distance += distance_vector[1]**2
		if self.include_z:
			distance += distance_vector[2]**2
		distance = math.sqrt(distance)
		d = max(1, distance / self._start_distance)
		data.append(d)

		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed
	
	# set start distance to goal to normalize by
	def reset(self, state=None):
		_drone_position = np.array(state['drone_position'])
		self._goal_position = np.array(state['goal_position'])
		distance_vector = self._goal_position - _drone_position
		self._start_distance = np.linalg.norm(distance_vector)