# trains a reinforcment learning algorithm
from controllers.controller import Controller
from component import _init_wrapper
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import time

class TrainRL(Controller):
    # constructor
    @_init_wrapper
    def __init__(self, model_component=''):
        super().__init__()

    # runs control on components
    def run(self):
        print('LEARN')
        self._model.learn()
        print('EVAL')
        self._model.evaluate()
        print('SAVE')
        self._model.save()