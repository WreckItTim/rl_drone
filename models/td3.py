# abstract class used to handle RL model
from models.model import Model
from component import _init_wrapper
import rl_utils as utils

class TD3(Model):
	# constructor
	@_init_wrapper
	def __init__(self,

			# parent params
			train_environment_component,
			actor,
			actor_target,
			critics,
			critics_target,
			obs_shape,
			act_shape,
			write_dir,
			save_init_model = True,
			save_init_buffer = False,
			replay_buffer = None, # str to read from file, None for new, numpy for exact
			buffer_size = 1_000_000, # number of recent samples to keep in replaybuffer
			device = 'cpu', # torch device, i.e. cpu or gpu
			obs_dtype = 'float32',
			act_dtype = 'float32',
			rew_dtype = 'float32',
			slim = 1, # current slim factor of actor (1 is full)
			end_buffer = 0, # index of last filled row in replay buffer
			rev_buffer = 0, # revolving index of next row to fill in replaybuffer
			nTrain = 0, # number of total train iters
			nSteps = 0, # number of sampling steps
			nEpisodes = 0, # number of sampling episodes

			# child params
			tau = 0.005, # polyak update coeff
			gamma = 0.99, # discount factor
			policy_delay = 2, # train iter delay netween updates
			noise_std = 0.2, # standard deviation of added action noise during train
			noise_max = 0.5, # max abs value of added action noise during train
		):
		self._is_hyper = False
		super().__init__()

	# adapted base code from psuedo @ https://spinningup.openai.com/en/latest/algorithms/td3.html
	def train(self,
			nIters=1,
			batch_size=100,
			with_distillation=False,
			low=0.125, size=2, # distill params
		):

		# UNSLIM (if no net modules are slim, then does nothing)
		for module in self.actor.modules():
			if 'Slim' in str(type(module)):
				module.slim = 1

		# do nIters many updates
		for itr in range(nIters):
			# sample replay buffer (actions are clones)
			obs, obs_next, act, rew, end = self.sample_buffer(batch_size)

			# sample next data from neural nets but do not calculate gradients
			with torch.no_grad():
				# sample next actions
				act_next = self.actor_target(obs_next)
				# add noise to next actions
				noise = torch.normal(0, self.noise_std, size=act_next.size())
				noise = torch.clamp(noise, min=-1*self.noise_max, max=self.noise_max)
				act_next = torch.clamp(act_next + noise, min=-1, max=1)
				# sample next q-vals and calculate target
				q_next = self.critic(obs_next, act_next, return_min=True)
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
				# update weights in critic
				critic.optimizer.step()

			self.nTrain += 1
			# update actor and target networks?
			if self.nTrain % self.policy_delay == 0:
				# compute actor loss w.r.t. first critic
				obs_act = torch.cat([obs, self.actor(obs)], 1)
				# maximize mean q-value from batch
                actor_loss = -self._critics[0](obs_act).mean()
				# calc gradient in actor
				self.actor.optimizer.zero_grad()
				actor_loss.backward(retain_graph=True)
				# DISTILL?? (if training to slim)
				if with_distillation:
					p = self.actor(replay_data.observations)
					sample_slim = np.random.uniform(low=low, high=1, size=size)
					slim_samples = [low] + list(sample_slim)
					for slim in slim_samples:
						for module in self.actor.modules():
							if 'Slim' in str(type(module)):
								module.slim = slim
						p2 = self.actor(obs)
						loss = F.mse_loss(p2, p)
						loss.backward(retain_graph=True)
					# UNSLIM
					for module in self.actor.modules():
						if 'Slim' in str(type(module)):
							module.slim = 1
				# update weights of actor
				self.actor.optimizer.step()
				# polyak updates on target networks
				self.polyak_update(self._actor.parameters(), self._actor_target.parameters(), self.tau)
				for c in range(self._nCritics):
					self.polyak_update(self._critics[c].parameters(), self._critics_target[c].parameters(), self.tau)
				
		# RESET SLIM - to value set from previous action
		for module in self.actor.modules():
			if 'Slim' in str(type(module)):
				module.slim = self.slim