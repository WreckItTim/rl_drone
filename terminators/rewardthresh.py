# penalizes colliding with objects
from terminators.terminator import Terminator
from component import _init_wrapper

class RewardThresh(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 rewarder_component,
                 min_reward=0,
                 ):
        super().__init__()

    # check for collision
    def terminate(self, state):
        if 'total_reward' not in state:
            self._rewarder.reward(state)
        if state['total_reward'] < self.min_reward:
            state['termination_reason'] = 'min_reward'
            return True
        return False