# abstract class used to handle sensors
from component import Component

class Sensor(Component):
    # constructor
    def __init__(self):
        pass

    # fetch a response from sensor
    def sense(self):
        raise NotImplementedError

    def activate(self):
        observation = self.sense()
        observation.display()

    def connect(self):
        super().connect()

    # creates a new observation object from passed in data
    def create_observation(self, data):
        raise NotImplementedError
