# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import TD3 as sb3TD3
from component import _init_wrapper

class TD3(Model):
	# constructor
	@_init_wrapper
	def __init__(self,
			environment_component,
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
			init_setup_model = False,
			model_path = None,
			replay_buffer_path = None,
		):
		kwargs = locals()
		_model_arguments = {key:kwargs[key] for key in kwargs.keys() if key not in [
			'self', 
			'__class__',
			'environment_component',
			'init_setup_model',
			'model_path',
			'replay_buffer_path',
			]}
		_model_arguments['_init_setup_model'] = kwargs['init_setup_model']
		self.sb3Type = sb3TD3
		self.sb3Load = sb3TD3.load
		self._has_replay_buffer = True
		super().__init__(model_path=model_path, 
				   replay_buffer_path=replay_buffer_path, 
				   _model_arguments=_model_arguments,
				   )