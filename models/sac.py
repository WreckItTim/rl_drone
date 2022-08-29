# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import SAC as sb3SAC
from component import _init_wrapper

class SAC(Model):
    # constructor
    @_init_wrapper
    def __init__(self,
            environment_component,
            policy = 'CnnPolicy',
            learning_rate = 1e-3,
            buffer_size = 1_000_000,
            learning_starts = 100,
            batch_size = 256,
            tau = 0.005,
            gamma = 0.99,
            train_freq = 1,
            gradient_steps = 1,
            action_noise = None,
            replay_buffer_class = None,
            replay_buffer_kwargs = None,
            optimize_memory_usage = False,
            ent_coef = "auto",
            target_update_interval = 1,
            target_entropy = "auto",
            use_sde = False,
            sde_sample_freq = -1,
            use_sde_at_warmup = False,
            tensorboard_log = None,
            create_eval_env = False,
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            init_setup_model = False,
            write_path = None,
            replay_buffer_path = None,
        ):
        kwargs = locals()
        _model_arguments = {key:kwargs[key] for key in kwargs.keys() if key not in ['self', '__class__', 'environment_component', 'init_setup_model', 'write_path', 'replay_buffer_path']}
        model_arguments['_init_setup_model'] = kwargs['init_setup_model']
        self.sb3Type = sb3SAC
        super().__init__(load_from_path=load_from_path, model_arguments=model_arguments)