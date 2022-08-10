# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np

class Point(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, drone_name='', xyz_point=[], min_distance=1, name=None):
        super().__init__()
        self.xyz_point = np.array(xyz_point, dtype=float)

    # checks if within distance of point
    def evaluate(self, state):
        if 'drone_position' not in state:
            state['drone_position'] = self._drone.get_position()
        drone_position = state['drone_position']
        distance = np.linalg.norm(drone_position - self.xyz_point)
        return distance <= self.min_distance

    def test(self):
        print(f'moving away from goal...')
        self._drone.move(0, 0, 20, 4, front_facing=True)
        print(f'position:{self._drone.get_position()} reached goal?:{self.evaluate({})}')
        print(f'moving toward goal...')
        self._drone.move(0, 0, -30, 4, front_facing=True)
        print(f'position:{self._drone.get_position()} reached goal?:{self.evaluate({})}')
