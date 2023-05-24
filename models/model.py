# abstract class used to handle RL model
from component import Component
import rl_utils as utils
import os
import numpy as np
import torch

# Generalized actor/critic model with replay buffer
# must define train() from child model
# see td3.py file for parent vs child params to pass into constructor
# buffer element at index -1 and 0th should be same element that follows index -2
	# this is for quicker sampling
class Model(Component):
	def __init__(self):
		self.write_dir = utils.fix_directory(self.write_dir)

	def connect(self, state=None):
		super().connect(state)

		# setup actor networks
		if type(self.actor) is str:
			self._actor = torch.load(self.actor)
		else:
			self._actor = self.actor
		if type(self.actor_target) is str:
			self._actor_target = torch.load(self.actor_target)
		else:
			self._actor_target = self.actor_target
		# setup critic networks
		if type(self.critics[0]) is str:
			self._critics = []
			for c in self.critics:
				self._critics.append(torch.load(c))
		else:
			self._critics = self.critics
		if type(self.critics_target[0]) is str:
			self._critics_target = []
			for c in self.critics_target:
				self._critics_target.append(torch.load(c))
		else:
			self._critics_target = self.critics_target
		self._nCritics = len(self._critics)
		if len(self._critics_target) != self._nCritics:
			utils.error('number of critics neq numner of target critics')
		# save init models
		self.actor = self.write_dir + 'actor.pt'
		self.actor_target = self.write_dir + 'actor_target.pt'
		self.critics = [self.write_dir + 'critics_' + str(i) for i in range(self._nCritics)]
		self.critics_target = [self.write_dir + 'critics_target_' + str(i) for i in range(self._nCritics)]
		if self.save_init_model:
			self.save_models(self.write_dir + 'model_init/')
		
		# setup replay buffer
		# create new from scratch
		if self.replay_buffer is None:
			self._replay_buffer = {
				'obs':np.zeros((self.buffer_size, *self.obs_shape), dtype=float),
				'act':np.zeros((self.buffer_size, *self.act_shape), dtype=float),
				'rew':np.zeros((self.buffer_size, 1), dtype=float),
				'end':np.zeros((self.buffer_size, 1), dtype='uint8'),
			}
		else:
			# read from path
			if type(self.replay_buffer) is str:
				self._replay_buffer = np.load(self.replay_buffer, allow_pickle=True)
			# save if passed in buffer obj (WARNING: not deepcopy - save mem)
			else:
				self._replay_buffer = self.replay_buffer
			# resize buffer as needed 
			for key in self._replay_buffer:
				old = self._replay_buffer[key].shape[0]
				if old > self.buffer_size:  # keep first elements that fit
					self._replay_buffer[key] = self._replay_buffer[:self.buffer_size,:]
				if old < self.buffer_size:
					new_shape = list(self._replay_buffer[key].shape)
					new_shape[0] = self.buffer_size
					self._replay_buffer[key].resize(new_shape, refcheck=False)
		# save init buffer
		self.replay_buffer = self.write_dir + 'replay_buffer.npz'
		if self.save_init_buffer:
			self.save_replay_buffer(self.write_dir + 'model_init/')

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
	
	# adds a single sample (step) to replay buffer
	def add_buffer(self, obs, act, rew, end):
		self._replay_buffer['obs'][self.rev_buffer] = obs.copy()
		self._replay_buffer['act'][self.rev_buffer] = act.copy()
		self._replay_buffer['rew'][self.rev_buffer] = rew
		self._replay_buffer['end'][self.rev_buffer] = end
		self.rev_buffer += 1
		if self.rev_buffer >= self.buffer_size:
			# (replicate values at end that to also be 0th element, for sampling)
			for key in self._replay_buffer:
				self._replay_buffer[key][0, :] = self._replay_buffer[key][-1, :]
			self.rev_buffer = 1
		self.end_buffer = min(self.end_buffer+1, self.buffer_size-1)

	# returns obs, next_obs, act (clone), rew, end
	# returns as tensors since typically called from train
	def sample_buffer(self, batch_size):
		idxs = np.random.randint(low=0, high=self.end_buffer, size=batch_size)
		return(
			torch.as_tensor(self._replay_buffer['obs'][idxs, :], device=self.device),
			torch.as_tensor(self._replay_buffer['obs'][idxs+1, :], device=self.device),
			torch.as_tensor(self._replay_buffer['act'][idxs, :].copy(), device=self.device),
			torch.as_tensor(self._replay_buffer['rew'][idxs, :], device=self.device),
			torch.as_tensor(self._replay_buffer['end'][idxs, :], device=self.device),
		)
	
	# SAVE METHODS
	# this will toggle if to checkpoint model and replay buffer from modifiers
	def set_save(self,
			  track_save,
			  track_vars=[
				  'model', 
				  'replay_buffer',
				  ],
			  ):
		self._track_save = track_save
		self._track_vars = track_vars.copy()
	# save torch models and replay_buffer to path
		# pass in write_folder to state
	def save(self, state):
		folder = utils.fix_directory(state['write_folder'])
		if 'model' in self._track_vars:
			self.save_models(folder)
		if 'replay_buffer' in self._track_vars:
			self.save_replay_buffer(folder)
	# save torch models to path
	def save_models(self, folder):
		folder = utils.fix_directory(folder)
		if not os.path.exists(folder):
			os.makedirs(folder)
		torch.save(self._actor, folder + 'actor.pt')
		torch.save(self._actor_target, folder + 'actor_target.pt')
		for i in range(self._nCritics):
			torch.save(self._critics[i], folder + 'critic_' + str(i) + '.pt')
			torch.save(self._critics_target[i], folder + 'critic_target_' + str(i) + '.pt')
	def load_models(self, folder):
		self._actor = torch.load(folder + 'actor.pt')
		self._actor_target = torch.load(folder + 'actor_target.pt')
		for i in range(self._nCritics)
			self._critics[i] = torch.load(folder + 'critic_' + str(i) + '.pt')
			self._critics_target[i] = torch.load(folder + 'critic_target_' + str(i) + '.pt')

	# save replay_buffer to path
	def save_replay_buffer(self, folder):
		np.savez(folder + 'replay_buffer.npz', **self._replay_buffer)

	# makes a prediction on best action given single observation
	# handles array (since typically called from env)
	def predict(self, observation):
		with torch.no_grad():
			tensor_in = torch.as_tensor(observation, device=self.device)
			tensor_out = self._actor(tensor_in)
			action = tensor_out.cpu().numpy()
		return action

	# makes an estimate on q-values given tensor of obserations and actions
	# handles tensor (since typically called from train)
	def critic(self, observations, actions, return_min=True, target=False):
		# concat rl_input and actions
		features = torch.cat([observations, actions], 1)
		# take min of all q-values
		q_vals = torch.zeros((len(features), self._nCritics))
		for c in range(self._nCritics):
			if target:
				q_vals[:, c] = self._critics_target[c](features).flatten()
			else:
				q_vals[:, c] = self._critics[c](features).flatten()
		if return_min:
			return torch.min(q_vals, dim=1, keepdim=True)[0]
		return q_vals

	# weighted combination of current and target params
	# tau of 1 is no polyak, just sets to target params
	def polyak_update(self, currents, targets, tau=1):
		with torch.no_grad():
			for current, target in zip(currents, targets):
				target.data.mul_(1 - tau)
				torch.add(target.data, current.data, alpha=tau, out=target.data)

	# called at begin of each episode
	def start(self, state = None):
		# reset slim factor to use full super-net
		for module in self._actor.modules():
			if 'Slim' in str(type(module)):
				module.slim = 1
		self._slim = 1

	# new learning loop
	def reset_learning(self, state=None):
		self.nTrain = 0 # number of total train iters
		self.nSteps = 0 # number of sampling steps
		self.nEpisodes = 0 # number of sampling episodes

	# runs learning loop on model
	def learn(self, 
			train_environment,
			max_episodes = 10_000,
			train_start = 100, # don't call train() until after train_start episodes
			train_freq = 1, # then call train() every train_freq episode
			batch_size = 100, # split training into mini-batches of steps from buffer
			with_distillation = False, # slims during train() and distills to output of super
			use_wandb = True, # turns on logging to wandb
			project_name = 'void', # project name in wandb
			# evaluation params
			evaluator = None,
		):
		# learning loop
		for _ in range(self.nEpisodes, max_episodes+1):
			# check evaluate
			evaluator.update()
			# start episode
			observation_data, state = train_environment.start()
			done = False
			episode_steps = 0
			while(not done):
				# get rl output
				rl_output = self.predict(observation_data)
				# take next step
				observation_data, reward, done, state = train_environment.step(rl_output)
				# log data to replay buffer
				self.add_buffer(observation_data, rl_output, reward, done)
				self.nSteps += 1
				episode_steps += 1
			# end of episode
			self.nEpisodes += 1
			# check train
			if self.nEpisodes >= train_start and self.nEpisodes % train_freq == 0:
				self.train(episode_steps, batch_size, with_distillation)