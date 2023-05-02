# abstract class used to handle RL model
from component import Component
import rl_utils as utils
import pickle
import os
import torch
import torch_layers as tl

class Model(Component):
	def __init__(self):
		self._is_slim = False
		self._slim = 1

	def connect(self):
		super().connect()
		
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

	def connect(self):
		super().connect()
		if self._is_hyper:
			return
		# create model object
		if self.read_path is not None and os.path.exists(self.read_path):
			utils.speak(f'reading model from path {self.read_path}')
			self.load_model(self.read_path)
			utils.speak(f'loaded torch from file')
			self.load_replay_buffer(self.read_path)
			utils.speak(f'loaded replay buffer from file')
		# convert all linear modules to slim ones
		if self.convert_slim:
			self._actor = tl.convert_to_slim(self._actor)
		# use slim layers
		if self.use_slim:
			self._is_slim = True
			# use distiallation with slim training
			if self.with_distillation:
				self.train = functools.partial(self.train_with_distillation, self)
		# set device for torch
		if self.device == 'cuda':
			device = torch.device('cuda')
			self._actor.cuda()
			self._actor_target.cuda()
			for critic in self._critics:
				critic.cuda()
			for critic in self._critics_target:
				critic.cuda()
		if self.device == 'cpu':
			device = torch.device('cpu')
			self._actor.cpu()
			self._actor_target.cpu()
			for critic in self._critics:
				critic.cpu()
			for critic in self._critics_target:
				critic.cpu()
		# save init model to file
		self.save_model(utils.get_global_parameter('working_directory') + 'model_init')
	
	# save sb3 model and replay_buffer to path
	# pass in write_folder to state
	def save(self, state):
		folder = utils.fix_directory(state['write_folder'])
		if 'model' in self._track_vars:
			self.save_model(folder)
		if 'replay_buffer' in self._track_vars:
			self.save_replay_buffer(folder)
	def save_model(self, folder):
		folder = utils.fix_directory(folder)
		if not os.path.exists(folder):
			os.makedirs(folder)
		torch.save(self._actor.state_dict(), folder + 'actor.pt')
		torch.save(self._actor_target.state_dict(), folder + 'actor_target.pt')
		for i in range(len(self._critics)):
			torch.save(self._critics[i].state_dict(), folder + 'critic_' + str(i) + '.pt')
			torch.save(self._critics_target[i].state_dict(), folder + 'critic_target_' + str(i) + '.pt')
	def save_replay_buffer(self, folder):
		pickle.dump(self._replay_buffer, open(folder + 'replay_buffer.p', 'wb'))

	# load sb3 model from path, must set sb3Load from child
	def load_model(self, folder):
		folder = utils.fix_directory(folder)
		self._actor = torch.load(folder + 'actor.pt')
		self._actor_target = torch.load(folder + 'actor_target.pt')
		self._critics = []
		self._critics_target = []
		i = 0
		while(True):
			if not os.path.exists(folder + 'critic_' + str(i) + '.pt'):
				break
			self._critics.append(torch.load(folder + 'critic_' + str(i) + '.pt'))
			self._critics_target.append(torch.load(folder + 'critic_target_' + str(i) + '.pt'))
			i += 1

	# load sb3 replay buffer from path
	def load_replay_buffer(self, folder):
		folder = utils.fix_directory(folder)
		self._replay_buffer = pickle.load(open(folder + 'replay_buffer.p', 'rb'))

	# makes a single prediction given input data
	def predict(self, rl_input):
		rl_output = self._actor(rl_input)
		return rl_output

	# reset slim factors
	def start(self, state = None):
		if self._is_slim:
			for module in self._sb3model.actor.modules():
				if 'Slim' in str(type(module)):
					module.slim = 1
			self._slim = 1

	# runs learning loop on model
	def learn(self, 
		total_timesteps=10_000,
		use_wandb = True,
		log_interval = -1,
		reset_num_timesteps = False,
		evaluator=None,
		project_name = 'void',
		):
		raise NotImplementedError

	# runs learning loop on model
	def train(self):
		raise NotImplementedError