# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np
import math

class Goal(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 drone_component, 
                 goal_component,
                 min_distance=5, 
                 max_distance=99999, 
                 include_z=True,
                 ):
        super().__init__()

    # checks if within distance of point
    def terminate(self, state):
        _drone_position = self._drone.get_position()
        _goal_position = self._goal.get_position()
        if not self.include_z:
            _drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
            _goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
        distance = np.linalg.norm(_drone_position - _goal_position)
        if distance < self.min_distance:
            state['termination_reason'] = 'min_distance'
            state['termination_result'] = 'success'
            return True
        if distance > self.max_distance:
            state['termination_reason'] = 'max_distance'
            state['termination_result'] = 'failure'
            return True
        return False