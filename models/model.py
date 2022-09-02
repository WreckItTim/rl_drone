# abstract class used to handle RL model
from component import Component
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import utils
from os.path import exists
from component import _init_wrapper

class Model(Component):
    # WARNING: child init must set sb3Type, and should have any child-model-specific parameters passed through model_arguments
    # NOTE: env=None as training and evaluation enivornments are handeled by controller
    def __init__(self, write_path=None, replay_buffer_path=None, _model_arguments=None):
        self._model_arguments = _model_arguments
        # set up write path
        if write_path is None:
            self.write_path = utils.global_parameters['write_folder'] + 'model'
        # set up replay buffer path
        if replay_buffer_path is None:
            self.replay_buffer_path = utils.global_parameters['write_folder'] + 'replay_buffer'
        self._sb3model = None
        self.connect_priority = -1 # environment needs to connect first if creating a new sb3model

    def connect(self):
        super().connect()
        # wrap environment for sb3
        #wrapped_environment = VecTransposeImage(DummyVecEnv([lambda: Monitor(self._environment)]))
        #self._model_arguments['env'] = wrapped_environment
        self._model_arguments['env'] = self._environment
        # create model object if needs be
        if self._sb3model is None:
            if self.write_path is not None and exists(self.write_path):
                self.load()
            else:
                self._sb3model = self.sb3Type(**self._model_arguments)
        if self.replay_buffer_path is not None and exists(self.replay_buffer_path):
            self._sb3model.load_replay_buffer(self.replay_buffer_path)

    def learn(self, 
        total_timesteps=1_000_000,
        callback = None,
        log_interval = -1,
        tb_log_name = None,
        eval_env = None,
        eval_freq = -1,
        n_eval_episodes = -1,
        eval_log_path = None,
        reset_num_timesteps = False,
        ):
        utils.speak('LEARN')
        # call sb3 learn method
        self._sb3model.learn(
            total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name = log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
        utils.speak('DONE LEARN')

    def save(self, path):
        self._sb3model.save(path)

    def save_replay_buffer(self, path):
        self._sb3model.save_replay_buffer(path)

    # loading is class specific - so must specify the stable-baselines3, or whatever, model type from child
    def load(self, path):
        if exists(path):
            utils.error(f'invalid Model.load() path:{path}')
        else:
            self._sb3model = self.sb3Type.load(path)

    # when using the debug controller
    def debug(self):
        self.learn()
        self.evaluate(self._environment)

    def evaluate(self,
        evaluate_environment,
        n_eval_episodes=1,
        deterministic=True, 
        render=False, 
        callback=None, 
        reward_threshold=None, 
        return_episode_rewards=False, 
        warn=False
        ):
        utils.speak('EVALUATE')
        evaluate_environment._evaluating = True
        # call sb3 evaluate method
        evaluate_policy(
            self._sb3model, 
            evaluate_environment, 
            n_eval_episodes=n_eval_episodes, 
            deterministic=deterministic, 
            render=render, 
            callback=callback, 
            reward_threshold=reward_threshold, 
            return_episode_rewards=return_episode_rewards, 
            warn=warn
        )
        evaluate_environment._evaluating = False
        utils.speak('DONE EVALUATE')