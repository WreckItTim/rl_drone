# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import TD3 as sb3TD3
from component import _init_wrapper

class TD3(Model):
    # constructor
    @_init_wrapper
    def __init__(self, 
            train_environment_component = '',
            evaluate_environment_component = '',
            policy = 'CnnPolicy',
            learning_rate = 1e-3,
            buffer_size = 1_000_000,
            learning_starts = 100,
            batch_size = 100,
            tau = 0.005,
            gamma = 0.99,
            train_freq = (1, "episode"),
            gradient_steps = -1,
            action_noise = None,
            replay_buffer_class = None,
            replay_buffer_kwargs = None,
            optimize_memory_usage = False,
            policy_delay = 2,
            target_policy_noise = 0.2,
            target_noise_clip = 0.5,
            tensorboard_log = None,
            create_eval_env = False,
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            init_setup_model = True,
        ):
        super().__init__()
        self._sb3model = sb3TD3(
            self.policy,
            self._train_environment,
            self.learning_rate,
            self.buffer_size,
            self.learning_starts,
            self.batch_size,
            self.tau,
            self.gamma,
            self.train_freq,
            self.gradient_steps,
            self.action_noise,
            self.replay_buffer_class,
            self.replay_buffer_kwargs,
            self.optimize_memory_usage,
            self.policy_delay,
            self.target_policy_noise,
            self.target_noise_clip,
            self.tensorboard_log,
            self.create_eval_env,
            self.policy_kwargs,
            self.verbose,
            self.seed,
            self.device,
            self.init_setup_model,
        )