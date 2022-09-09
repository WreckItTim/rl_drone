# abstract class used to handle custom data structures
from component import Component

class DataStruct(Component):
    # constructor
    def __init__(self):
        pass

    def connect(self):
        super().connect()

    def reset(self):
        pass

    def step(self, state):
        pass