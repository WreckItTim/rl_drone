# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np
import math

class RelativePoint(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 drone_component, 
                 xyz_point, 
                 min_distance=5, 
                 max_distance=99999, 
                 include_z=True,
                 ):
        super().__init__()
        self.xyz_point = np.array(xyz_point, dtype=float)
        self._x = self.xyz_point[0]
        self._y = self.xyz_point[1]
        self._z = self.xyz_point[2]

    # checks if within distance of point
    def terminate(self, state):
        _drone_position = self._drone.get_position()
        _xyz_point = self.xyz_point
        if not self.include_z:
            _drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
            _xyz_point = np.array([_xyz_point[0], _xyz_point[1]], dtype=float)
        distance = np.linalg.norm(_drone_position - _xyz_point)
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