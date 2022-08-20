# checks if drone has exceeded number of timesteps
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np

class MaxSteps(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, max_steps=100):
        super().__init__()
        self._nSteps = 0

    # checks if within distance of point
    def evaluate(self, state):
        self._nSteps += 1
        if self._nSteps > self.max_steps:
            return True
        return False

    def reset(self):
        self._nSteps = 0