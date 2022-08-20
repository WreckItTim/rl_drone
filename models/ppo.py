# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import PPO as sb3PPO
from component import _init_wrapper

class PPO(Model):
    # constructor
    @_init_wrapper
    def __init__(self, 
            train_environment_component = None,
            evaluate_environment_component = None,
            policy = 'CnnPolicy',
            learning_rate = 1e-3,
            n_steps = 1, #2048
            batch_size = 64,
            n_epochs = 10,
            gamma = 0.99,
            gae_lambda = 0.95,
            clip_range = 0.2,
            clip_range_vf = None,
            normalize_advantage = True,
            ent_coef = 0.0,
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            use_sde = False,
            sde_sample_freq = -1,
            target_kl = None,
            tensorboard_log = None,
            create_eval_env = False,
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            init_setup_model = True,
        ):
        super().__init__()
        self._sb3model = sb3PPO(
            self.policy,
            self._train_environment,
            self.learning_rate,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.gamma,
            self.gae_lambda,
            self.clip_range,
            self.clip_range_vf,
            self.normalize_advantage,
            self.ent_coef,
            self.vf_coef,
            self.max_grad_norm,
            self.use_sde,
            self.sde_sample_freq,
            self.target_kl,
            self.tensorboard_log,
            self.create_eval_env,
            self.policy_kwargs,
            self.verbose,
            self.seed,
            self.device,
            self.init_setup_model,
        )