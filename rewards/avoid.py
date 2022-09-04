# penalizes colliding with objects
from rewards.reward import Reward
from component import _init_wrapper

class Avoid(Reward):
    # constructor
    @_init_wrapper
    def __init__(self, drone_component):
        super().__init__()

    # -1 for a collision, +1 for dodging collision
    def reward(self, state):
        if 'has_collided' not in state:
            state['has_collided'] = self._drone.check_collision()
        has_collided = state['has_collided']
        value = 0
        if has_collided:
            value = -1
        return value