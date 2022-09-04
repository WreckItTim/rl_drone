# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np
import math

class RelativePoint(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, drone_component, xyz_point, min_distance=5, max_distance=99999):
        super().__init__()
        self.xyz_point = np.array(xyz_point, dtype=float)
        self._x = self.xyz_point[0]
        self._y = self.xyz_point[1]
        self._z = self.xyz_point[2]

    # checks if within distance of point
    def terminate(self, state):
        if 'drone_position' not in state:
            state['drone_position'] = self._drone.get_position()
        drone_position = np.array(state['drone_position'], dtype=float)
        if 'distance' not in state:
            state['distance'] = np.linalg.norm(drone_position - self.xyz_point)
        distance = state['distance']
        if distance < self.min_distance:
            state['termination_reason'] = 'min_distance'
            return True
        if distance > self.max_distance:
            state['termination_reason'] = 'max_distance'
            return True
        return False

    def reset(self):
        position = self._drone.get_position()
        yaw = self._drone.get_yaw() # yaw counterclockwise rotation about z-axis
        x = position[0] + self._x * math.cos(yaw) + self._y * math.sin(yaw)
        y = position[1] + self._y * math.cos(yaw) + self._x * math.sin(yaw)
        z = position[2] + self._z
        self.xyz_point = np.array([x, y, z], dtype=float)