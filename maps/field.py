# abstract class used to handle maps - where the drone is flying in
from maps.map import Map
from component import _init_wrapper

class Field(Map):
    # constructor
    @_init_wrapper
    def __init__(self):
        super().__init__()