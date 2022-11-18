# gets point distance from drone (set through airsim settings) on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
import utils
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
			  raw_code=None,
			  ):
		super().__init__(offline, raw_code)

	def create_obj(self, data):
		observation = Vector(
			_data=data, 
		)
		return observation

	# takes a picture with camera
	def sense2(self):
		distance = np.array(self._airsim._client.getDistanceSensorData().distance)
		observation = self.create_obj([distance])
		return observation