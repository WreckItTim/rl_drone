# penalizes colliding with objects
from terminators.terminator import Terminator
from component import _init_wrapper

class Collision(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, drone_component):
        super().__init__()

    # check for collision
    def terminate(self, state):
        if 'has_collided' not in state:
            state['has_collided'] = self._drone.check_collision()
        has_collided = state['has_collided']
        if has_collided:
            state['termination_reason'] = 'collided'
            state['termination_result'] = 'failure'
        return has_collided