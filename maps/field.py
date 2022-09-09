# abstract class used to handle maps - where the drone is flying in
from maps.map import Map
from component import _init_wrapper

class Field(Map):
    # constructor
    @_init_wrapper
    def __init__(self,
                 # voxels for 2d/3d numpy array represtation of objects
                 voxels_component=None,
                 ):
        super().__init__()