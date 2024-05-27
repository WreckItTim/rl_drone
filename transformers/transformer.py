# robots in disguise! Transforms an observation
from component import Component

class Transformer(Component):
    # constructor
    def __init__(self):
        pass

    # if observation type is valid, applies transformation
    def transform(self, observation):
        raise NotImplementedError

    def connect(self, state=None):
        super().connect(state)