# checks if drone has reached point
from terminators.terminator import Terminator
from component import _init_wrapper
import numpy as np
import math
import utils

class RelativePoint(Terminator):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 drone_component, 
                 map_component,
                 xyz_point, 
                 min_distance=5, 
                 max_distance=99999, 
                 include_z=True,
                 random_yaw=False,
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
            state['termination_result'] = 'success'
            return True
        if distance > self.max_distance:
            state['termination_reason'] = 'max_distance'
            state['termination_result'] = 'failure'
            return True
        return False

    def get_xyz(self, position, yaw, alpha):
        x = position[0] + alpha*self._x * math.cos(yaw) + alpha*self._y * math.sin(yaw)
        y = position[1] + alpha*self._y * math.cos(yaw) + alpha*self._x * math.sin(yaw)
        z = position[2] + self._z
        in_object = self._map.at_object_2d(x, y)
        return x, y, z, in_object
    
    # need to recalculate relative point at each reset
    def reset(self, reset_state):
        if 'goal' in reset_state:
            self.xyz_point = np.array(reset_state['goal'], dtype=float)
        else:
            position = self._drone.get_position()
            if self.random_yaw:
                yaw = random.uniform(0, 2*math.pi)
            else:
                yaw = self._drone.get_yaw()  # yaw counterclockwise rotation about z-axis
            # shorten the distance until not in object (this is a cheap trick, better to think about points first)
            alpha = 1
            in_object = True
            while in_object:
                x, y, z, in_object = self.get_xyz(position, yaw, alpha)
                alpha -= 0.1
                if alpha < 0.1:
                    utils.error('invalid objective point')
            reset_state['goal'] = [x, y, z]
            self.xyz_point = np.array([x, y, z], dtype=float)