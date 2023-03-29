# abstract class used to handle RL model
from component import Component
import rl_utils as utils
from os.path import exists
import wandb
import torch
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
import torch.nn as nn
from torch import Tensor
import copy

# CUSTOM SLIM LAYERS
class Slim(nn.Linear):
	def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True,
				 device=None, dtype=None,
				slim_in=True, slim_out=True) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super().__init__(max_in_features, max_out_features, bias, device, dtype)
		self.max_in_features = max_in_features
		self.max_out_features = max_out_features
		self.slim_in = slim_in
		self.slim_out = slim_out
		self.slim = 1
		
	def forward(self, input: Tensor) -> Tensor:
		if self.slim_in:
			self.in_features = int(self.slim * self.max_in_features)
		if self.slim_out:
			self.out_features = int(self.slim * self.max_out_features)
		weight = self.weight[:self.out_features, :self.in_features]
		if self.bias is not None:
			bias = self.bias[:self.out_features]
		else:
			bias = self.bias
		y = F.linear(input, weight, bias)
		#utils.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
		return y

# modifies a TD3 train() to add distillation for slimming
# ASSUMES custom Slim layers
def train_with_distillation(self, gradient_steps: int, batch_size: int = 100) -> None:
	utils.speak('BEGIN TRAIN')
	# Switch to train mode (this affects batch norm / dropout)
	self.policy.set_training_mode(True)

	# Update learning rate according to lr schedule
	self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

	# UNSLIM (code added from original SB3 train() method )
	actor_losses, critic_losses = [], []
	for module in self.actor.modules():
		if 'Slim' in str(type(module)):
			module.slim = 1
	for _ in range(gradient_steps):
		self._n_updates += 1
		# Sample replay buffer
		replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

		with th.no_grad():
			# Select action according to policy and add clipped noise
			noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
			noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
			next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

			# Compute the next Q-values: min over all critics targets
			next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
			next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
			target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

		# Get current Q-values estimates for each critic network
		current_q_values = self.critic(replay_data.observations, replay_data.actions)

		# Compute critic loss
		critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
		critic_losses.append(critic_loss.item())

		# Optimize the critics
		self.critic.optimizer.zero_grad()
		critic_loss.backward()
		self.critic.optimizer.step()

		# Delayed policy updates
		if self._n_updates % self.policy_delay == 0:
			# Compute actor loss
			actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
			actor_losses.append(actor_loss.item())

			# Optimize the actor
			self.actor.optimizer.zero_grad()
			actor_loss.backward(retain_graph=True)

			# DISTILL (code added from original SB3 train() method )
			p = self.actor(replay_data.observations)
			sample_slim = np.random.uniform(low=0.1, high=1, size=2)
			slim_samples = [0.1] + list(sample_slim)
			for slim in slim_samples:
				for module in self.actor.modules():
					if 'Slim' in str(type(module)):
						module.slim = slim
				p2 = self.actor(replay_data.observations)
				loss = criterion(p2, p)
				loss.backward(retain_graph=True)
			# UNSLIM
			for module in self.actor.modules():
				if 'Slim' in str(type(module)):
					module.slim = 1

			# step
			self.actor.optimizer.step()

			polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
			polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
			# Copy running stats, see GH issue #996
			polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
			polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

	self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
	if len(actor_losses) > 0:
		self.logger.record("train/actor_loss", np.mean(actor_losses))
	self.logger.record("train/critic_loss", np.mean(critic_losses))

	# RESET SLIM - to value set from previous action
	for module in self.actor.modules():
		if 'Slim' in str(type(module)):
			module.slim = self.slim
	utils.speak('END TRAIN')

# custom class that derives from SB3 - so that it adds noise to replay buffer
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

# custom class taht derives from tensor callback to add whatever you want to tb log
class TensorboardCallback(BaseCallback):
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

def convert_to_slim(model):
	#after calling, set as such: new_model = copy.deepcopy(model) ..
	nLinearLayers = 0
	for module in model.modules():
		if 'Linear' in str(type(module)):
			nLinearLayers += 1
	modules = []
	onLinear = 0
	for module in model.modules():
		if 'Sequential' in str(type(module)):
			continue
		elif 'Linear' in str(type(module)):
			onLinear += 1
			max_in_features = module.in_features
			max_out_features = module.out_features
			bias = module.bias is not None
			slim_in, slim_out = True, True
			if onLinear == 1:
				slim_in = False
			if onLinear == nLinearLayers:
				slim_out = False
			new_module = Slim(max_in_features, max_out_features,
							bias=bias, slim_in=slim_in, slim_out=slim_out)
			modules.append(new_module)
		else:
			modules.append(module)
	new_model = nn.Sequential(*modules)
	new_model.load_state_dict(copy.deepcopy(model.state_dict()))
	return new_model

class Model(Component):
	# WARNING: child init must set sb3Type, and should have any child-model-specific parameters passed through model_arguments
		# child init also needs to save the training environment (make environment_component a constructor parameter)
	# NOTE: env=None as training and evaluation enivornments are handeled by controller
	def __init__(self, 
				read_model_path=None, 
				read_replay_buffer_path=None, 
				read_weights_path=None, 
				_model_arguments=None,
				with_distillation = True,
				use_slim = False,
				convert_slim = False,
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
		self._sb3model.actor.optimizer = torch.optim.Adam(
			self._sb3model.actor.parameters(),
			amsgrad=False,
			betas= (0.9, 0.999),
			capturable= False,
			differentiable= False,
			eps= 1e-8,
			foreach= None,
			fused= False,
			lr= 1e-3,
			maximize= False,
			weight_decay= 1e-6,
		)
		self._sb3model.critic.optimizer = torch.optim.Adam(
			self._sb3model.critic.parameters(),
			amsgrad=False,
			betas= (0.9, 0.999),
			capturable= False,
			differentiable= False,
			eps= 1e-8,
			foreach= None,
			fused= False,
			lr= 1e-3,
			maximize= False,
			weight_decay= 1e-6,
		)
		# load weights?
		if self.read_weights_path is not None and exists(self.read_weights_path):
			self.load_weights(self.read_weights_path)
		# convert all linear modules to slim ones
		if self.convert_slim:
			self._sb3model.actor.mu = convert_to_slim(self._sb3model.actor.mu)
			self._sb3model.actor_target.mu = convert_to_slim(self._sb3model.actor_target.mu)
			self._sb3model.slim = 1
		# use slim layers
		if self.use_slim:
			self._sb3model.slim = 1
			# use distiallation with slim training
			if self.with_distillation:
				self._sb3model.train = train_with_distillation
		# save init model to file
		self.save_model(utils.get_global_parameter('working_directory') + 'model_in.zip')
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

	def save_weights(self, actor_path, critic_path):
		torch.save(self._sb3model.actor.state_dict(), actor_path)
		torch.save(self._sb3model.critic.state_dict(), critic_path)

	def load_weights(self, actor_path, critic_path):
		self._sb3model.actor.load_state_dict(torch.load(actor_path))
		self._sb3model.actor_target.load_state_dict(torch.load(actor_path))
		self._sb3model.critic.load_state_dict(torch.load(critic_path))
		self._sb3model.critic_target.load_state_dict(torch.load(critic_path))

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
			project="SECON23_epsilon",
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
		
	# makes a single prediction given input data
	def predict(self, rl_input):
		rl_output, next_state = self._sb3model.predict(rl_input, deterministic=True)
		return rl_output
