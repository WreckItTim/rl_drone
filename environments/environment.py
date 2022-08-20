# abstract class used to handle observations to input into rl algo
from component import Component, get_component

class Environment(Component):
    # constructor
    def __init__(self):
        pass

    def activate(self):
        return self.step(None)
        
    def step(self, rl_output):
        raise NotImplementedError

    # called at end of episode to prepare for next, when step() returns done=True
    # returns first observation for new episode
    def reset(self):
        raise NotImplementedError