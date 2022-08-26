# abstract class used to handle rewards in an RL enivornment
from component import Component

class Rewarder(Component):
    # constructor
    def __init__(self):
        pass

    # calculates rewards from agent's current state (call to when taking a step)
    def reward(self, state):
        raise NotImplementedError

    def activate(self):
        state = {}
        self.reward(state)
        return state

    def connect(self):
        super().connect()