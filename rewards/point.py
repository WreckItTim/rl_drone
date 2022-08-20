# rewards the closer the object is to the point
from rewards.reward import Reward
from component import _init_wrapper
import numpy as np

class Point(Reward):
    # constructor
    @_init_wrapper
    def __init__(self, drone_component='', xyz_point=[], min_distance=1, max_distance=1000):
        super().__init__()
        self.xyz_point = np.array(xyz_point, dtype=float)
        self._diff = max_distance - min_distance

    # -1 for a collision, +1 for dodging collision
    def evaluate(self, state):
        if 'drone_position' not in state:
            state['drone_position'] = self._drone.get_position()
        drone_position = state['drone_position']
        if 'distance' not in state:
            state['distance'] = np.linalg.norm(drone_position - self.xyz_point)
        distance = state['distance']
        if distance < self.min_distance:
            total_reward = 1
        elif distance > self.max_distance:
            total_reward = -1
        else: 
            # normalize between -1 furtherst, +1 closest
            total_reward = (-2)*((distance - self.min_distance)/self._diff) + 1
        return total_reward

    def test(self):
        print(f'comparing point reward to point:{self.xyz_point}...')
        print(f'position:{self._drone.get_position()} reward:{self.evaluate({})}')
        print(f'moving drone upward by {2*self.min_distance} units...')
        self._drone.move(0, 0, 2*self.min_distance, 4, front_facing=False)
        print(f'position:{self._drone.get_position()} reward:{self.evaluate({})}')
        print(f'moving drone upward by another {self.max_distance/2} units...')
        self._drone.move(0, 0, self.max_distance/2, 4, front_facing=False)
        print(f'position:{self._drone.get_position()} reward:{self.evaluate({})}')
