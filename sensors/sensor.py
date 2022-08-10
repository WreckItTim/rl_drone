# abstract class used to handle sensors
from component import Component

class Sensor(Component):
    # constructor
    def __init__(self):
        pass

    # fetch a response from sensor
    def sense(self):
        raise NotImplementedError