# penalizes colliding with objects
from terminators.terminator import Terminator
from component import _init_wrapper

class Collision(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, drone_component):
        super().__init__()

    # check for collision
    def evaluate(self, state):
        if 'has_collided' not in state:
            state['has_collided'] = self._drone.check_collision()
        has_collided = state['has_collided']
        return has_collided