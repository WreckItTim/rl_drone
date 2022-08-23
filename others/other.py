# abstract class used to handle abstract components
from component import Component

class Other(Component):
    def __init__(self):
        raise NotImplementedError

    def step(self, state):
        pass

    def reset(self):
        pass

    def activate(self):
        pass