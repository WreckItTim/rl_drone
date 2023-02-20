from controllers.controller import Controller
from component import _init_wrapper

# simply runs an evaluation set on the given evaluation environment
class Evaluate(Controller):
    # constructor
    @_init_wrapper
    def __init__(self, 
                 evaluator_component, 
                 ):
        super().__init__()

    # runs control on components
    def run(self):
        self._evaluator.evaluate_set()