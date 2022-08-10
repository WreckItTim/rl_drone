# abstract class used to handle maps - where the drone is flying in
from maps.map import Map
from component import _init_wrapper

class UCIField(Map):
    # constructor
    @_init_wrapper
    def __init__(self, name=None):
        super().__init__()