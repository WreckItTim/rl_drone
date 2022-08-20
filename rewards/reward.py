# abstract class used to handle rewards in an RL enivornment
from component import Component

class Reward(Component):
    # constructor
    def __init__(self):
        pass

    # calculates rewards from agent's current state (call to when taking a step)
    def evaluate(self, state):
        raise NotImplementedError

    def activate(self):
        return self.evaluate({})