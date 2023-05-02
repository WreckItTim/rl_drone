# abstract class used to handle RL model
from models.model import Model
from component import _init_wrapper

class TD3(Model):
	# constructor
	@_init_wrapper
	def __init__(self,
			_actor,
			_critics,
			_actor_target = None,
			_critics_target = None,
			actor_path = None,
			critics_path = None,
			environment_component,
			buffer_size = 1_000_000,
			learning_starts = 100,
			batch_size = 100,
			tau = 0.005,
			gamma = 0.99,
			train_freq = (1, "episode"),
			policy_delay = 2,
			target_policy_noise = 0.2,
			target_noise_clip = 0.5,
			device = "cpu",
			read_path=None,
			with_distillation = False,
			use_slim = False,
			convert_slim = False,
		):
		pass

	# runs learning loop on model
	def learn(self, 
		total_timesteps=10_000,
		use_wandb = True,
		log_interval = -1,
		reset_num_timesteps = False,
		evaluator=None,
		project_name = 'void',
		):
		
		for timestep in range(total_timesteps):
			# reset environment, returning first observation
			observation_data = self._train_environment.start()
			# start episode
			done = False
			while(not done):
				# get rl output
				rl_output = self.predict(observation_data)
				# take next step
				#observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
				observation_data, reward, done = self._evaluate_environment.step(rl_output)
			# call end for modifiers
			self.end()

	def train(self, gradient_steps, batch_size = 100):
		#utils.speak('BEGIN TRAIN with distill')
		#for param in self.actor_target.parameters():
		#	print(param.device)
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
				actor_loss = -self.critic.q1_forward(replay_data.oactor_targetbservations, self.actor(replay_data.observations)).mean()
				actor_losses.append(actor_loss.item())

				# Optimize the actor
				self.actor.optimizer.zero_grad()
				actor_loss.backward(retain_graph=True)

				# DISTILL (code added from original SB3 train() method )
				p = self.actor(replay_data.observations)
				sample_slim = np.random.uniform(low=0.1251, high=0.9999, size=2)
				slim_samples = [0.125] + list(sample_slim)
				print(f'distilling:{slim_samples}')
				for slim in slim_samples:
					for module in self.actor.modules():
						if 'Slim' in str(type(module)):
							module.slim = slim
					p2 = self.actor(replay_data.observations)
					loss = F.mse_loss(p2, p)
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
		#utils.speak('END TRAIN with distill')