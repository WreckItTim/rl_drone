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
        self._reward_function = lambda distance : math.exp(-1.0 * distance)
        self.init_normalization()

    # calculate constants for normalization
    def init_normalization(self):
        # get and min and max reward outputs from inputting min and max distances
        self.y_min = self._reward_function(self.min_distance)
        y_max = self._reward_function(self.max_distance)
        self.y_diff = y_max - self.y_min

    # normalize reward value between 0 and 1
    def normalize_reward(self, distance):
        # clip to min-max distance
        clipped_distance = min(self.max_distance, max(self.min_distance, distance))
        # get value from decaying exponential (heavier rewards for closer)
        y = self._reward_function(clipped_distance)
        # min-max normalize between 0 and 1
        return (y - self.y_min) / self.y_diff
    
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