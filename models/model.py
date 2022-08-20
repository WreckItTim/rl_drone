# abstract class used to handle RL model
from component import Component
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class Model(Component):
    # constructor
    def __init__(self):
        # wrap environments for sb3
        self._train_environment = VecTransposeImage(DummyVecEnv([lambda: Monitor(self._train_environment)]))
        self._evaluate_environment = VecTransposeImage(DummyVecEnv([lambda: Monitor(self._evaluate_environment)]))

    def learn(self, 
        total_timesteps=10,
        callback = None,
        log_interval = 1,
        eval_env = None,
        eval_freq = -1,
        n_eval_episodes = 5,
        eval_log_path = None,
        reset_num_timesteps = False,
        ):
        self._sb3model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def evaluate(self, 
        n_eval_episodes=1,
        deterministic=True, 
        render=False, 
        callback=None, 
        reward_threshold=None, 
        return_episode_rewards=False, 
        warn=True
        ):
        reward = evaluate_policy(
            self._sb3model, 
            self._evaluate_environment, 
            n_eval_episodes, 
            deterministic, 
            render, 
            callback, 
            reward_threshold, 
            return_episode_rewards, 
            warn
        )
        return reward

    def save(self, write_path='sb3_saved_model'):
        self._sb3model.save(write_path)