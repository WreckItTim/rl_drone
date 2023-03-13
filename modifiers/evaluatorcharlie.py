from modifiers.modifier import Modifier
from component import _init_wrapper
import rl_utils as utils
from configuration import Configuration
import numpy as np
import os

# Charlie works in goal based environments (A to B)
# Charlie evaluates every number episodes
# Charlie actively saves the best model (optinally at each epoch as well)
# Charlie amps up distance to goal on each success
	# then repeats for each success until failure
# Charlie will switch phase from distance to optimize reward
# Charlie will then switch to final phase from reward to increase noise level
# if the model doesn't improve after some episodes
	# then Charlie will end the learning loop
# Charlie updates _best_score in the model component for hyper training
class EvaluatorCharlie(Modifier):
	@_init_wrapper
	def __init__(self, 
				base_component, # componet with method to modify
				parent_method, # name of parent method to modify
				order, # modify 'pre' or 'post'?
				evaluate_environment_component, # environment to run eval in
				model_component, # used to make predictions, track best model
				noises_components, # list of compenents to amp up noise with curriculum learning
				bounds_component, # will change bounds with curriuclum learning
				best_distance = 0, # distance metric to maximize
				best_reward = -999_999, # reward metric to maximize
				best_noise = 0, # noise metric to maximize
				amp_up_static = [4, 0, 0], # increases static goal distance
				amp_up_random = 4, # increases random goal distance
				phase = 'distance', # toggle which phase to evaluate at
				distance_max = 100, # will train until succesfully reaches goal this far away
				reward_epsilon = 1e-2, # will train until reward does not improve by this much
				reward_patience = 10, # will train until does not reach success within this many evals
				noise_start = 10, # how many epochs to wait to start patience (collect noisy data first)
				noise_patience = 10, # will train until does not reach success within this many evals
				wait = 0, # keeps track of in-house
				success = -1, # how many successfull episodes it takes to trigger an all_success
				nEpisodes = 1, # number of episodes to evaluate each set
				set_counter = 0, # count number of eval sets
				save_every_model = False, # will save each model at each epoch
				write_folder = None, # writes best model / replay buffer here
				track_vars = ['model'], # which best vars to write
				verbose = 1, # handles output (level 1 is set level, level 2 is episode level)
				on_evaluate = True, # toggle to run modifier on evaluation environ
				on_train = True, # toggle to run modifier on train environ
				frequency = 1, # use modifiation after how many calls to parent method?
				counter = 0, # keepts track of number of calls to parent method
				activate_on_first = True, # will activate on first call otherwise only if % is not 0
				current_noise = 0, # keeps track of current level of noise being applied to sensors 
				): 
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)
		self.connect_priority = -2 # needs other components to connect first
		self.amp_up_static = np.array(amp_up_static, dtype=float)
		if success < 0:
			self.success = self.nEpisodes

	def connect(self, state=None):
		super().connect(state)
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._name + '/'
			if not os.path.exists(self.write_folder):
				os.makedirs(self.write_folder)
		# give noise component visiblity of progress in learning loop
		if self._model._model_arguments['action_noise'] is not None:
			self._model._model_arguments['action_noise'].set_progress_calculator(self)

	# returns progress of learning loop
	def get_progress(self):
		return self.best_distance / self.distance_max

	def activate(self, state=None):
		if self.check_counter(state):
			# TEMPORARY CODE TO CHECK IS TRAINING
			# print first weight (just to check if training)
			'''
			model_name = str(self._model._child())
			sb3_model = self._model._sb3model
			if 'dqn' in model_name:
				for name, param in sb3_model.q_net.named_parameters():
					msg = str(name) + ' ____ ' + str(param[0])
					utils.speak(msg)
					break
			if 'td3' in model_name:
				for name, param in sb3_model.critic.named_parameters():
					msg = str(name) + ' ____ ' + str(param[0])
					utils.speak(msg)
					break
			'''
			# permanent code below
			# evaluate for a set of episodes, until failure
			while(True):
				stop, another_set = self.evaluate_set()
				if not another_set:
					break
			# close up shop?
			if stop:
				self._configuration.controller.stop()

	# reset learning loop to static values from connect()
	def reset_learning(self):
		self.best_distance = -1
		self.best_reward = -999_999
		self.current_noise = 0
		self.best_noise = 0
		self.wait = 0
		self.phase = 'distance'
		self._model._best_score = 0

	# steps through one evaluation episode
	def evaluate_episode(self):
		# reset environment, returning first observation
		observation_data = self._evaluate_environment.reset()
		# start episode
		done = False
		total_reward = 0
		while(not done):
			# get rl output
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
			total_reward += reward
		# call end for modifiers
		self._model.end()
		# end of episode
		return state['termination_result'] == 'success', total_reward

	# evaluates all episodes for this next set
	def evaluate_set(self):

		# keep track of episode results
		total_success = 0
		total_reward = 0
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			this_success, this_reward = self.evaluate_episode()
			total_success += this_success
			total_reward += this_reward
		all_success = total_success >= self.success
		mean_reward = total_reward / self.nEpisodes

		if self.verbose > 0:
			utils.speak(f'Evaluation:{self.set_counter} distance:{self._evaluate_environment._goal.random_dim_min} reward:{round(mean_reward,2)} noise:{self.current_noise} nSuccess:{total_success}')

		# save every model
		if self.save_every_model:
			self._model.save_model(self.write_folder + 'model_' + str(self.set_counter) + '.zip')

		stop = False # used to determine if we stop OUTTER TRAINING LOOP - typically shuts down entire program
		another_set = False # used to determine if another set should run	
		# only accept potential evaluations if all_success was triggered
		if all_success:
			new_best = False # measures if we improved from last epochs

			# are we optimizing goal distance?
			if self.phase == 'distance':
				new_best = True # as long as we reached all_success then we are improving
				this_distance = self._evaluate_environment._goal.random_dim_min
				self._model._best_score = -1 * this_distance # need to minimize if hyper learning
				self.best_distance = this_distance
				if this_distance >= self.distance_max:
					# save final best here (otherwise will be lost when switching phases)
					if 'model' in self.track_vars:
						self._model.save_model(self.write_folder + 'best_model_distance.zip')
					if 'replay_buffer' in self.track_vars:
						self._model.save_replay_buffer(self.write_folder + 'best_replay_buffer_distance.zip')
					# switch to reward phase
					self.phase = 'reward'
					self.best_reward = mean_reward
					self._model._best_score = -100 * mean_reward # need to minimize if hyper learning
					if self.verbose > 0:
						utils.speak(f'Switching eval phase from distance to reward...')
				else:
					# amp up goal distance
					self._evaluate_environment._goal.amp_up(self.amp_up_static, self.amp_up_random, self.amp_up_random)
					self._bounds.apply_delta_inner([self.amp_up_random, self.amp_up_random, 0])
					another_set = True # evaluate again, to see if we can get even farther without more training

			# are we optimizing reward?
			elif self.phase == 'reward':
				if (mean_reward - self.best_reward) > self.reward_epsilon:
					new_best = True
					self.best_reward = mean_reward
					self._model._best_score = -100 * mean_reward # need to minimize if hyper learning
				else:
					self.wait += 1
				if self.wait > self.reward_patience:
					# switch to noise phase
					self.phase = 'noise'
					self._model._best_score = -10_000 # need to minimize if hyper learning
					# start wait to an offset to allow us to collect noisy data 
					self.wait = -1 * self.noise_start
					# amp up noise levels 
					for _noise in self._noises:
						_noise.amp_up_noise()
					self.current_noise = 1
					if self.verbose > 0:
						utils.speak(f'Switching eval phase from reward to noise...')

			# are we optimizing noise?
			elif self.phase == 'noise':
				new_best = True # as long as we reached all_success then we are improving
				another_set = True # evaluate again, to see if we can achieve goal with more noise
				self.best_noise += 1
				self._model._best_score = -10_000 * (self.best_noise) # need to minimize if hyper learning
				# amp up noise levels 
				for _noise in self._noises:
					_noise.amp_up_noise()
				self.current_noise += 1

			# do we update best epoch?
			if new_best:
				# save best
				if 'model' in self.track_vars:
					self._model.save_model(self.write_folder + 'best_model_' + self.phase + '.zip')
				if 'replay_buffer' in self.track_vars:
					self._model.save_replay_buffer(self.write_folder + 'best_replay_buffer_' + self.phase + '.zip')
				# update early stopping
				self.wait = min(self.wait, 0) # consider any start as offset (i.e. noise_start)
				if self.verbose > 0:
					utils.speak(f'New best! best_distance:{self.best_distance} best_reward:{round(self.best_reward,2)} best_noise:{self.best_noise}')
		else:
			# update early stopping
			self.wait += 1

		# check to terminate if wait is too high
		if self.phase == 'noise' and self.wait > self.noise_patience:
			if self.verbose > 0:
				utils.speak(f'reached noise patience, stopping training loop...')
			stop = True
			
		self.set_counter += 1
		return stop, another_set
