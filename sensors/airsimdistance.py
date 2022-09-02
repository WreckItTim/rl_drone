# gets point distance from drone (set through airsim settings) on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
from observations.distance import Distance
import numpy as np
from component import _init_wrapper

class AirSimDistance(Sensor):

    # constructor
    @_init_wrapper
    def __init__(self):
        super().__init__()
        self._client = None

    # resets on episode
    def reset(self):
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

    def connect(self):
        super().connect()
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()

    # takes a picture with camera
    def sense(self):
        data = np.array(self._client.getDistanceSensorData().distance)
        distance = Distance(
            _data=data, 
        )
        return distance

    # creates a new observation object from passed in data
    def create_observation(self, data):
        distance = Distance(
            _data=data, 
        )
        return distance