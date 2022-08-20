# penalizes colliding with objects
from terminators.terminator import Terminator
from component import _init_wrapper

class RewardThresh(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, min_reward=0):
        super().__init__()

    # check for collision
    def evaluate(self, state):
        done = False
        if state['total_reward'] < self.min_reward:
            done = true
        return done