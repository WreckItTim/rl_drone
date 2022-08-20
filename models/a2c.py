# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import A2C as sb3A2C
from component import _init_wrapper

class A2C(Model):
    # constructor
    @_init_wrapper
    def __init__(self, 
            train_environment_component = '',
            evaluate_environment_component = '',
            policy = 'CnnPolicy',
            learning_rate = 7e-4,
            n_steps = 5,
            gamma = 0.99,
            gae_lambda = 1.0,
            ent_coef = 0.0,
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            rms_prop_eps = 1e-5,
            use_rms_prop = True,
            use_sde = False,
            sde_sample_freq = -1,
            normalize_advantage = False,
            tensorboard_log = None,
            create_eval_env = False,
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            init_setup_model = True,
        ):
        super().__init__()
        self._sb3model = sb3A2C(
            self.policy,
            self._train_environment,
            self.learning_rate,
            self.n_steps,
            self.gamma,
            self.gae_lambda,
            self.ent_coef,
            self.vf_coef,
            self.max_grad_norm,
            self.rms_prop_eps,
            self.use_rms_prop,
            self.use_sde,
            self.sde_sample_freq,
            self.normalize_advantage,
            self.tensorboard_log,
            self.create_eval_env,
            self.policy_kwargs,
            self.verbose,
            self.seed,
            self.device,
            self.init_setup_model,
        )