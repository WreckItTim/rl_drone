from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np

# keeps track of previous action
# self normalizes to [0,1]
class Actions(Sensor):
	
	@_init_wrapper
	def __init__(self,
                 actor_component,
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
	def step(self, state):
		nSteps = state['nSteps']
		if nSteps == 0:
			if self._actor._type == 'discrete':
				data = [0]
			else:
				data = [0] * len(self._actor._actions)
		else:	
			rl_output = state['rl_output']
			if self._actor._type == 'discrete':
				# add 1 to length so that 0 is reserved for no-data
				data = [rl_output / (len(self._actor._actions)+1)]
			else:
				# subtract by epsilon so that 0 is reserved for no-data
				epsilon = 1e-2
				data = [0] * len(rl_output)
				for i in range(len(rl_output)):
					data[i] = (rl_output[i] - self._actor._actions[i].min_space + epsilon) / (self._actor._actions[i].max_space - self._actor._actions[i].min_space + epsilon)

		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed