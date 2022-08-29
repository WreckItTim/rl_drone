# rewards the closer the object is to the point
from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math

class RelativePoint(Reward):
    # constructor
    @_init_wrapper
    def __init__(self, drone_component, xyz_point, min_distance=5, max_distance=100):
        super().__init__()
        self.xyz_point = np.array(xyz_point, dtype=float)
        self._diff = max_distance - min_distance
        self._x = self.xyz_point[0]
        self._y = self.xyz_point[1]
        self._z = self.xyz_point[2]

    # -1 for a collision, +1 for dodging collision
    def reward(self, state):
        if 'drone_position' not in state:
            state['drone_position'] = self._drone.get_position()
        drone_position = np.array(state['drone_position'], dtype=float)
        if 'distance' not in state:
            state['distance'] = float(np.linalg.norm(drone_position - self.xyz_point))
        distance = state['distance']
        if distance < self.min_distance:
            total_reward = 1
        elif distance > self.max_distance:
            total_reward = -1
        else: 
            # normalize between -1 furtherst, +1 closest
            total_reward = (-2)*((distance - self.min_distance)/self._diff) + 1
        return total_reward

    def reset(self):
        position = self._drone.get_position()
        yaw = self._drone.get_yaw(radians=True) # yaw counterclockwise rotationa bout z-axis
        x = position[0] + self._x * math.cos(yaw) + self._y * math.sin(yaw)
        y = position[1] + self._y * math.cos(yaw) + self._x * math.sin(yaw)
        z = position[2] + self._z
        self.xyz_point = np.array([x, y, z], dtype=float)