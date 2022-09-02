from datastructs.datastruct import DataStruct
from component import _init_wrapper
import random
import numpy as np

# data structure specifying a rectangluar-prism zone
class Zone(DataStruct):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 x_min:int=0, 
                 x_max:int=0, 
                 y_min:int=0, 
                 y_max:int=0, 
                 z_min:int=0, 
                 z_max:int=0,
                 ):
        super().__init__()

    # checks bounds if point is inside prism, point is np array [x,y,z] (include boundary)
    def in_bounds(self, point):
        if point[0] >= self.x_min and point[0] <= self.x_max:
            if point[1] >= self.y_min and point[1] <= self.y_max:
                if point[2] >= self.z_min and point[2] <= self.z_max:
                    return True
        return False

    # get a random point inside prism
    def random_point(self):
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        return np.array([x, y, z], dtype=float)

    # sample several random points inside prism
    def sample_points(self, nPoints):
        points = [None] * nPoints
        for i in range(nPoints):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            z = random.uniform(self.z_min, self.z_max)
            points[i] = np.array([x, y, z], dtype=float)
        return points

    # debug mode
    def debug(self):
        return self.random_point()