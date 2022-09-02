from datastructs.datastruct import DataStruct
from component import _init_wrapper
import random
import numpy as np

# data structure specifying a spawn zone
class Spawn(DataStruct):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 x=0,
                 y=0,
                 z=0,
                 yaw=0,
                 x_min=0, 
                 x_max=0, 
                 y_min=0, 
                 y_max=0, 
                 z_min=0, 
                 z_max=0,
                 yaw_min=0,
                 yaw_max=0,
                 random=True,
                 ):
        super().__init__()
        if random:
            self.get_spawn = self.random_spawn
        else:
            self.get_spawn = self.static_spawn
        
    def random_spawn(self):
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        yaw = random.uniform(self.yaw_min, self.yaw_max)
        return [x, y, z], yaw
        
    def static_spawn(self):
        return [self.x, self.y, self.z], self.yaw

    # debug mode
    def debug(self):
        return self.get_spawn()