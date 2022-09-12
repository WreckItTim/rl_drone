from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math

# calculates distance between drone and point relative to starting position/orientation
class Goal(Reward):
    # constructor, set the relative point and min-max distances to normalize by
    @_init_wrapper
    def __init__(self,
                 drone_component, 
                 goal_component, 
                 min_distance, 
                 max_distance, 
                 goal_tolerance=0,
                 include_z=True,
                 ):
        super().__init__()
        # set reward function
        #self._reward_function = lambda x : math.exp(-2.0 * x)
        #self._reward_function = lambda x : 1-x
        #self._reward_function = lambda x : 1-x
        self.init_normalization()

    # calculate constants for normalization
    def init_normalization(self):
        # normalize to min and max distances
        self._diff = self.max_distance - self.min_distance

    # normalize reward value between 0 and 1
    def normalize_reward(self, distance):
        # clip distance
        clipped_distance = max(self.min_distance, min(self.max_distance, distance))
        # normalize distance to fit desired behavior of reward function
        normalized_distance = (clipped_distance - self.min_distance) / self._diff
        # get value from reward function
        value = self._reward_function(normalized_distance)
        return value
    
    # get reward based on distance to point 
    def reward(self, state):
        _drone_position = self._drone.get_position()
        _goal_position = self._goal.get_position()
        if not self.include_z:
            _drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
            _goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
        distance = np.linalg.norm(_drone_position - _goal_position)
        print('distance_to_goal', distance)
        if distance <= self.goal_tolerance:
            return 1
        return 0
        #value = self.normalize_reward(distance)
        #return value 