# abstract class used to handle all components
from component import Component

class Controller(Component):
    # constructor
    def __init__(self):
        pass

    # runs control on components
    def run(self):
        raise NotImplementedError