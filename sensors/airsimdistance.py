# gets point distance from drone (set through airsim settings) on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
import rl_utils as utils
from observations.vector import Vector
import numpy as np
from component import _init_wrapper

class AirSimDistance(Sensor):

	# constructor
	@_init_wrapper
	def __init__(self,
			  airsim_component,
			  transformers_components=None,
			  offline = False,
			  ):
		super().__init__(offline)

	def create_obj(self, data):
		observation = Vector(
			_data=data, 
		)
		return observation

	# takes a picture with camera
	def step(self, state=None):
		distance = np.array(self._airsim._client.getDistanceSensorData().distance)
		observation = self.create_obj([distance])
		transformed = self.transform(observation)
		return transformed