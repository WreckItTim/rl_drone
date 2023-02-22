from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np

# keeps track of step / max steps
# makes state fully observable if using steps rewards
class Steps(Sensor):
	
	@_init_wrapper
	def __init__(self,
                 steps_component,
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
		# get dynamic max steps from reward component
		max_steps = self._steps._max_steps 
		# calculate ratio of current steps to max
		data = [nSteps / max_steps]

		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed