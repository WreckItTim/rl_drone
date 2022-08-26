# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import A2C as sb3A2C
from component import _init_wrapper
from os.path import exists

class A2C(Model):
    # constructor
    @_init_wrapper
    def __init__(self, 
            environment_component,
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
            init_setup_model = False,
            write_path = None,
        ):
        kwargs = locals()
        model_arguments = {key:kwargs[key] for key in kwargs.keys() if key not in ['self', '__class__', 'environment_component', 'init_setup_model', 'write_path']}
        model_arguments['_init_setup_model'] = kwargs['init_setup_model']
        self.sb3Type = sb3A2C
        super().__init__(load_from_path=load_from_path, model_arguments=model_arguments)