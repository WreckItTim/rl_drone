# abstract class used to handle actions
from actions.action import Action
from component import Component

class Actor(Component):
    # constructor
    def __init__(self):
        pass

    # interprets output from an RL model to take action, then takes it
    def act(self, rl_output=None):
        raise NotImplementedError
        
    def activate(self):
        self.act()