from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math

# calculates distance between drone and point relative to starting position/orientation
class RelativePoint(Reward):
    # constructor, set the relative point and min-max distances to normalize by
    @_init_wrapper
    def __init__(self, drone_component, xyz_point, min_distance, max_distance):
        super().__init__()
        self.xyz_point = np.array(xyz_point, dtype=float)
        self._x = self.xyz_point[0]
        self._y = self.xyz_point[1]
        self._z = self.xyz_point[2]
        # set reward function
        self._reward_function = lambda x : math.exp(-2.0 * x)
        self.init_normalization()

    # calculate constants for normalization
    def init_normalization(self):
        # normalize to min and max distances
        self._diff = self.max_distance - self.min_distance

    # normalize reward value between 0 and 1
    def normalize_reward(self, distance):
        # clip to min_distance so reward does not go over 1
        clipped_distance = max(self.min_distance, distance)
        # normalize distance to fit desired behavior of reward function
        normalized_distance = (clipped_distance - self.min_distance) / self._diff
        # get value from reward function
        value = self._reward_function(normalized_distance)
        return value
    
    # get reward based on distance to point 
    def reward(self, state):
        if 'drone_position' not in state:
            state['drone_position'] = self._drone.get_position()
        drone_position = np.array(state['drone_position'], dtype=float)
        if 'distance' not in state:
            state['distance'] = float(np.linalg.norm(drone_position - self.xyz_point))
        distance = state['distance']
        value = self.normalize_reward(distance)
        return value

    # need to recalculate relative point at each reset
    def reset(self):
        position = self._drone.get_position()
        yaw = self._drone.get_yaw() 
        x = position[0] + self._x * math.cos(yaw) + self._y * math.sin(yaw)
        y = position[1] + self._y * math.cos(yaw) + self._x * math.sin(yaw)
        z = position[2] + self._z
        self.xyz_point = np.array([x, y, z], dtype=float)