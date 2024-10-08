# abstract class used to handle RL model
from custommodels.custommodel import CustomModel
from component import _init_wrapper
import rl_utils as utils
import torch
import pickle

class TD3(CustomModel):
	# constructor
	@_init_wrapper
	def __init__(self,
			write_dir,
			obs_shape=None,
			act_shape=None,
			actor=None,
			actor_target=None,
			critics=None,
			critics_target=None,
			save_init_model = False,
			save_init_buffer = False,
			replay_buffer = None, # str to read from file, None for new, numpy for exact
			buffer_size = 1_000_000, # number of recent samples to keep in replaybuffer
			device = 'cpu', # torch device, i.e. cpu or gpu
			end_buffer = 0, # index of last filled row in replay buffer
			rev_buffer = 0, # revolving index of next row to fill in replaybuffer
			nTrain = 0, # counter for total train iters
			nSteps = 0, # counter for train steps
			nEpisodes = 0, # counter for train episodes
			tau = 0.005, # polyak update coeff
			gamma = 0.99, # discount factor
			policy_delay = 2, # train iter delay netween updates
			noise_std = 0.2, # standard deviation of added action noise during train
			noise_max = 0.5, # max abs value of added action noise during train
			explore_std = 0.1, # standard deviation of added action noise during train
		):
		super().__init__()

	# adapted base code from psuedo @ https://spinningup.openai.com/en/latest/algorithms/td3.html
	def train(self,
			batch_size=100,
			num_batches=1,
		):
		self.set_train()
		
		# do nIters many updates
		for batch in range(num_batches):
			# sample replay buffer (actions are clones)
			obs, obs_next, act, rew, end, idxs = self.sample_buffer(batch_size)

			# sample next data from neural nets but do not calculate gradients
			with torch.no_grad():
				# sample next actions
				act_next = self._actor_target(obs_next)
				# add noise to next actions
				noise = torch.normal(0, self.noise_std, size=act_next.size())
				noise = torch.clamp(noise, min=-1*self.noise_max, max=self.noise_max)
				act_next2 = torch.clamp(act_next + noise, min=-1, max=1)
				# sample next q-vals and calculate target
				q_next = self.critic(obs_next, act_next2, return_min=True, target=True)
				q_target = rew + (1 - end) * self.gamma * q_next

			# update each critic
			obs_act = torch.cat([obs, act], 1)
			for critic in self._critics:
				# calc q values from current observations and actions
				q_vals = critic(obs_act)
				# calc critic loss
				critic_loss = torch.nn.functional.mse_loss(q_vals, q_target)
				# calc gradient in critic
				critic.optimizer.zero_grad()
				critic_loss.backward()
				critic.optimizer.step()
				
			self.nTrain += 1
			# update actor and target networks?
			if self.nTrain % self.policy_delay == 0:
				# compute actor loss w.r.t. first critic
				obs_act = torch.cat([obs, self._actor(obs)], 1)
				# maximize mean q-value from batch
				actor_loss = -self._critics[0](obs_act).mean()
				# calc gradient in actor
				self._actor.optimizer.zero_grad()
				actor_loss.backward()
				# update weights of actor
				self._actor.optimizer.step()
				# polyak updates on target networks
				self.polyak_update(self._actor.parameters(), self._actor_target.parameters(), self.tau)
				for c in range(self._nCritics):
					self.polyak_update(self._critics[c].parameters(), self._critics_target[c].parameters(), self.tau)
		
		self.set_eval()