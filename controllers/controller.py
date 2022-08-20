# abstract class used to handle all components
from component import Component

class Controller(Component):
    # constructor
    def __init__(self):
        self._add_to_list = False
        self._name = 'controller'

    # runs control on components
    def run(self):
        raise NotImplementedError