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
import functools
from numpy.typing import DTypeLike

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
		#print(f'B4-shape:{self.weight.shape}')
		weight = self.weight[:self.out_features, :self.in_features]
		if self.bias is not None:
			bias = self.bias[:self.out_features]
		else:
			bias = self.bias
		y = F.linear(input, weight, bias)
		#utils.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
		return y

# convert a neural network model to slimmable
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
				use_slim = False,
				convert_slim = False,
			):
		self._model_arguments = _model_arguments
		self._sb3model = None
		self.connect_priority = -1 # environment needs to connect first if creating a new sb3model
		self._is_slim = False

	def connect(self):
		super().connect()
		self._model_arguments['env'] = self._environment
		# read sb3 model from file
		if self.read_model_path is not None and exists(self.read_model_path):
			self.load_model(self.read_model_path,
				tensorboard_log = self._model_arguments['tensorboard_log'],
			)
			self._sb3model.set_env(self._model_arguments['env'])
			self._sb3model.learning_starts = self._model_arguments['learning_starts']
			self._sb3model.train_freq = self._model_arguments['train_freq']
			self._sb3model._convert_train_freq()
			utils.speak('loaded model from file')
		# create sb3 model from scratch
		else:
			self._sb3model = self.sb3Type(**self._model_arguments)
		# convert all linear modules to slim ones
		if self.convert_slim:
			self._is_slim = True
			self._sb3model.actor.mu = convert_to_slim(self._sb3model.actor.mu)
			#self._sb3model.actor_target.mu = convert_to_slim(self._sb3model.actor_target.mu)
			self._sb3model.slim = 1
			utils.speak('converted model to slimmable')
		# use slim layers
		if self.use_slim:
			self._is_slim = True
			self._sb3model.slim = 1
			utils.speak('using slimmable model')
		# read replay buffer from file
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
		self._sb3model.save(path)
	def save_replay_buffer(self, path):
		self._sb3model.save_replay_buffer(path)

	# load sb3 model from path, must set sb3Load from child
	def load_model(self, path, tensorboard_log=None):
		if not exists(path):
			utils.error(f'invalid Model.load_model() path:{path}')
		else:
			self._sb3model = self.sb3Load(path, tensorboard_log=tensorboard_log)

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
		total_timesteps = 10_000,
		log_interval = -1,
		reset_num_timesteps = False,
		tb_log_name = None,
		use_wandb = False,
		wandb_project_name = 'void',
		):
		callback = None
		if use_wandb:
			wandb_config = {
				"policy_type": self.policy,
				"total_timesteps": total_timesteps,
			}
			run = wandb.init(
				project = wandb_project_name,
				config = wandb_config,
				name = utils.get_local_parameter('run_name'),
				sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
				monitor_gym=False,  # auto-upload the videos of agents playing the game
				save_code=False,  # optional
			)
			callback = [WandbCallback(gradient_save_freq=100)]
		# call sb3 learn method
		self._sb3model.learn(
			total_timesteps,
			callback = callback,
			log_interval = log_interval,
			tb_log_name = tb_log_name,
			reset_num_timesteps = reset_num_timesteps,
		)
		if use_wandb:
			run.finish()
		
	# makes a single prediction given input data
	def predict(self, rl_input):
		rl_output, next_state = self._sb3model.predict(rl_input, deterministic=True)
		return rl_output

	# reset slim factors
	def reset(self, state = None):
		if self._is_slim:
			for module in self._sb3model.actor.modules():
				if 'Slim' in str(type(module)):
					module.slim = 1
			self._sb3model.slim = 1
