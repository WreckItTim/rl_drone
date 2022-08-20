# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import DQN as sb3DQN
from component import _init_wrapper

class DQN(Model):
    # constructor
    @_init_wrapper
    def __init__(self, 
            train_environment_component = '',
            evaluate_environment_component = '',
            policy = 'CnnPolicy',
            learning_rate = 1e-4,
            buffer_size = 1_000_000,
            learning_starts = 50000,
            batch_size = 32,
            tau = 1.0,
            gamma = 0.99,
            train_freq = 4,
            gradient_steps = 1,
            replay_buffer_class = None,
            replay_buffer_kwargs = None,
            optimize_memory_usage = False,
            target_update_interval = 10000,
            exploration_fraction = 0.1,
            exploration_initial_eps = 1.0,
            exploration_final_eps = 0.05,
            max_grad_norm = 10,
            tensorboard_log = None,
            create_eval_env = False,
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            init_setup_model = True,
        ):
        super().__init__()
        self._sb3model = sb3DQN(
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
            self.replay_buffer_class,
            self.replay_buffer_kwargs,
            self.optimize_memory_usage,
            self.target_update_interval,
            self.exploration_fraction,
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.max_grad_norm,
            self.tensorboard_log,
            self.create_eval_env,
            self.policy_kwargs,
            self.verbose,
            self.seed,
            self.device,
            self.init_setup_model,
        )