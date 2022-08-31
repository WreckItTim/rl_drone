# trains a reinforcment learning algorithm
from controllers.controller import Controller
from component import _init_wrapper
import time

class TrainRL(Controller):
    # constructor
    @_init_wrapper
    def __init__(self, model_component):
        super().__init__()

    # runs control on components
    def run(self, set_save_to_path=True):
        self._model.learn()