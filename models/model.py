# abstract class used to handle RL model
from component import Component
import rl_utils as utils
from os.path import exists
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback

# derives from SB3 - so they add noise to buffer
class NormalActionNoise2(ActionNoise):
	def __init__(self, 
	mean = 0, 
	sigma = 0.05, 
	clip = 0.2, 
	exploration_fraction = 0.5, 
	start_proba = 1.0, 
	end_propa = 0.05,
	static_proba = 0.5,
	):
		self._mu = mean
		self._sigma = sigma
		self._clip = clip
		self._fraction = exploration_fraction # fraction of progress to reach end proba
		self._start = start_proba # proba to add noise at start on learning loop
		self._end = end_propa # final propa to add noise after fraction of progress 
			# _end < _start
		self._scale = (self._end  - self._start) / self._fraction # scale proba by progress
		self._noise = 0
		self._progress_calculator = None
		self._static_proba = static_proba
		super().__init__()

	# progress_calculator is an arbitrary object with a get_progress() method
	# progress returns [0,1] and is progress towards terminating learning loop
	def set_progress_calculator(self, progress_calculator):
		self._progress_calculator = progress_calculator

	def __call__(self):
		self._noise = 0
		proba = self._static_proba
		if self._progress_calculator is not None: 
			# get progress in learning loop [0,1]
			# 0 is no progress, 1 is complete
			progress = self._progress_calculator.get_progress()
			if progress > self._fraction:
				proba = self._end
			else:
				proba =  self._start + progress * self._scale
		if np.random.uniform() < proba:
			self._noise = max(-1*self._clip, (min(self._clip, 
				np.random.normal(self._mu, self._sigma)
			)))
		return self._noise

	def __repr__(self) -> str:
		return f"NormalActionNoise2(mu={self._mu}, sigma={self._sigma})"

class TensorboardCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""
	def __init__(self, evaluator, verbose=0):
		super().__init__(verbose)
		self.evaluator = evaluator
	def _on_step(self) -> bool:
		best_distance = self.evaluator.best_distance
		self.logger.record("best_distance", best_distance)
		best_reward = self.evaluator.best_reward
		self.logger.record("best_reward", best_reward)
		best_noise = self.evaluator.best_noise
		self.logger.record("best_noise", best_noise)

class Model(Component):
	# WARNING: child init must set sb3Type, and should have any child-model-specific parameters passed through model_arguments
		# child init also needs to save the training environment (make environment_component a constructor parameter)
	# NOTE: env=None as training and evaluation enivornments are handeled by controller
	def __init__(self, 
			  read_model_path=None, 
			  read_replay_buffer_path=None, 
			  _model_arguments=None
			  ):
		# if the model is a hyper parameter tuner, some things get handeled differently
		self._is_hyper = False
		if _model_arguments['action_noise'] == 'normal':
			_model_arguments['action_noise'] = NormalActionNoise2()
		self._model_arguments = _model_arguments
		self._sb3model = None
		self.connect_priority = -1 # environment needs to connect first if creating a new sb3model

	def connect(self):
		super().connect()
		if self._is_hyper:
			return
		self._model_arguments['env'] = self._environment
		# create model object
		if self.read_model_path is not None and exists(self.read_model_path):
			utils.speak(f'reading model from path {self.read_model_path}')
			self.load_model(self.read_model_path)
			self._sb3model.set_env(self._model_arguments['env'])
			utils.speak('loaded model from file')
		else:
			self._sb3model = self.sb3Type(**self._model_arguments)
		# replay buffer init
		if self.read_replay_buffer_path is not None and exists(self.read_replay_buffer_path):
			self.load_replay_buffer(self.read_replay_buffer_path)
			utils.speak('loaded replay buffer from file')
		
			
	# this will toggle if to checkpoint model and replay buffer
	def set_save(self,
			  track_save,
			  track_vars=[
				  'model', 
				  'replay_buffer',
				  ],
			  ):
		self._track_save = track_save
		self._track_vars = track_vars.copy()
	
	# save sb3 model and replay_buffer to path
	# pass in write_folder to state
	def save(self, state):
		write_folder = state['write_folder']
		if 'model' in self._track_vars:
			self.save_model(write_folder + 'model.zip')
		if 'replay_buffer' in self._track_vars and self._has_replay_buffer:
			self.save_replay_buffer(write_folder + 'replay_buffer.zip')
	def save_model(self, path):
		# SB3 has built in serialization which can not handle a custom class
		if self._sb3model.action_noise is not None:
			temp = self._sb3model.action_noise._progress_calculator
			self._sb3model.action_noise._progress_calculator = None
		self._sb3model.save(path)
		# SB3 has built in serialization which can not handle a custom class
		if self._sb3model.action_noise is not None:
			self._sb3model.action_noise._progress_calculator = temp
	def save_replay_buffer(self, path):
		self._sb3model.save_replay_buffer(path)

	# load sb3 model from path, must set sb3Load from child
	def load_model(self, path):
		if not exists(path):
			utils.error(f'invalid Model.load_model() path:{path}')
		else:
			self._sb3model = self.sb3Load(path)
		
	# load sb3 replay buffer from path
	def load_replay_buffer(self, path):
		if not exists(path):
			utils.error(f'invalid Model.load_replay_buffer() path:{path}')
		elif self._has_replay_buffer:
			self._sb3model.load_replay_buffer(path)
		else:
			utils.warning(f'trying to load a replay buffer to a model that does not use one')

	# runs learning loop on model
	def learn(self, 
		total_timesteps=10_000,
		use_wandb = True,
		log_interval = -1,
		reset_num_timesteps = False,
		evaluator=None,
		):
		config = {
			"policy_type": self.policy,
			"total_timesteps": total_timesteps,
		}
		run = wandb.init(
			project="sb3",
			config=config,
			name = utils.get_global_parameter('run_name'),
			sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
			monitor_gym=False,  # auto-upload the videos of agents playing the game
			save_code=False,  # optional
		)
		callback = None
		if use_wandb:
			callback = [
				TensorboardCallback(evaluator),
				WandbCallback(
					gradient_save_freq=100,
					),
			]
		# call sb3 learn method
		self._sb3model.learn(
			total_timesteps,
			callback=callback,
			log_interval= log_interval,
			tb_log_name = 'tb_log',
			reset_num_timesteps = reset_num_timesteps,
		)
		run.finish()
		
	# makes a single prediction g;iven input data
	def predict(self, rl_input):
		rl_output, next_state = self._sb3model.predict(rl_input, deterministic=True)
		return rl_output
