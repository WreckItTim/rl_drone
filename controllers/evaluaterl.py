# trains a reinforcment learning algorithm
from controllers.controller import Controller
from component import _init_wrapper

class EvaluateRL(Controller):
    # constructor
    @_init_wrapper
    def __init__(self, model_component):
        super().__init__()

    # runs control on components
    def run(self):
        print('EVALUATE')
        self._model.evaluate(self._model.environment)