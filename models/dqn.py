# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import DQN as sb3DQN
from component import _init_wrapper

class DQN(Model):
	# constructor
	@_init_wrapper
	def __init__(self, 
			environment_component,
			policy = 'MlpPolicy',
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
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			read_model_path=None, 
			read_replay_buffer_path=None, 
		):
		kwargs = locals()
		_model_arguments = {key:kwargs[key] for key in kwargs.keys() if key not in [
			'self', 
			'__class__',
			'environment_component',
			'init_setup_model',
			'read_model_path',
			'read_replay_buffer_path',
			]}
		_model_arguments['_init_setup_model'] = kwargs['init_setup_model']
		self.sb3Type = sb3DQN
		self.sb3Load = sb3DQN.load
		self._has_replay_buffer = True
		super().__init__(
				   read_model_path=read_model_path, 
				   read_replay_buffer_path=read_replay_buffer_path, 
				   _model_arguments=_model_arguments,
				   )