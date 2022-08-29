# trains a reinforcment learning algorithm
from controllers.controller import Controller
from component import _init_wrapper

class EvaluateRL(Controller):
    # constructor
    @_init_wrapper
    def __init__(self, model_component, n_eval_episodes=1):
        super().__init__()

    # runs control on components
    def run(self):
        self._model.evaluate(self._model._environment, n_eval_episodes=self.n_eval_episodes)