# abstract class used to handle observations to input into rl algo
from component import Component
from os.path import exists
from os import makedirs

class Observer(Component):
    # constructor
    def __init__(self, please_write=False, write_directory=None):
        if please_write:
            if not exists(write_directory):
                makedirs(write_directory)

    # processes observation after captured by child
    # returns as numpy for rl handling
    def process(self, observation):
        if self.please_write:
            observation.write(self.write_directory)
        return observation.to_numpy()
            

    # returns observation transcribed for input into RL model
    def observe(self):
        raise NotImplementedError