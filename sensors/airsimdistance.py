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
			  transformers_components=None,
			  offline = False,
			  ):
		super().__init__(offline)
		self._client = None

	# resets on episode
	def reset(self):
		self._client.enableApiControl(True)
		self._client.armDisarm(True)

	def connect(self):
		super().connect()
		self._client = airsim.MultirotorClient(
			ip=utils.get_global_parameter('LocalHostIp'),
			port=utils.get_global_parameter('ApiServerPort'),
										 )
		self._client.confirmConnection()

	# takes a picture with camera
	def sense(self):
		distance = np.array(self._client.getDistanceSensorData().distance)
		observation = Vector(
			_data=[distance], 
		)
		return self.transform(observation)