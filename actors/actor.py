# abstract class used to handle actions
from actions.action import Action
from component import Component

class Actor(Component):
    # constructor
    def __init__(self):
        pass

    # interprets output from an RL model to take action, then takes it
    # returns transcribed action as string (for logging purposes only)
    def act(self, rl_output):
        raise NotImplementedError

    def connect(self):
        super().connect()

    def reset(self):
        for action in self._actions:
            action.reset()

    # returns action space for this actor
    def get_space(self):
        raise NotImplementedError