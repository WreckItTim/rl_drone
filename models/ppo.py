# abstract class used to handle RL model
from models.model import Model
from stable_baselines3 import PPO as sb3PPO
from component import _init_wrapper

class PPO(Model):
	# constructor
	@_init_wrapper
	def __init__(self, 
			environment_component,
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
		self.sb3Type = sb3PPO
		self.sb3Load = sb3PPO.load
		self.has_replay_buffer = False
		super().__init__(model_path=model_path, 
				   replay_buffer_path=replay_buffer_path, 
				   _model_arguments=_model_arguments,
				   )