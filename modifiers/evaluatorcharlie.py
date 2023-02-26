from modifiers.modifier import Modifier
from component import _init_wrapper
import utils
from configuration import Configuration
import numpy as np
import os

# Charlie works in goal based environments (A to B)
# Charlie evaluates every number episodes
# Charlie actively saves the best model
# Charlie amps up distance to goal on each success
	# then repeats for each success until failure
# if the model doesn't improve after some episodes
	# then Charlie will end the learning loop
# Charlie updates _best_score in the model component
class EvaluatorCharlie(Modifier):
	@_init_wrapper
	def __init__(self, 
				base_component, # componet with method to modify
				parent_method, # name of parent method to modify
				order, # modify 'pre' or 'post'?
				evaluate_environment_component, # environment to run eval in
				model_component, # used to make predictions, track best model
				best_score = 0, # metric to improve
				best_eval = 0, # evaluation set corresponding to best_score
				best_counter = 0, # counter corresponding to first evaluation that lead to best_score
				amp_up_static = [4, 0, 0], # increases static goal distance
				amp_up_random = 4, # increases random goal distance
				stop_amp_at = 92, # will stomp amping up when goal random_dim_min is this large
				amping_phase = True, # toggles to false when done amping up then works on reward
				nEpisodes = 1, # number of episodes to evaluate each set
				success = -1, # number of successfull episodes for set success, -1=all
				patience = 999999, # number of sets to wait to improve best_score
				epsilon_order = 2, # order of magnitude to evaluate early stopping (must improve by this many orders)
					# from our paper implementation, an epsilon_order between 2-4 was near an improvement of 1 step per episode
				wait = 0, # number of sets have been waiting to improve score
				set_counter = 0, # count number of eval sets
				random = False, # set true to select randomly from spawn objects
				write_folder = None, # writes best model / replay buffer here
				track_vars = ['model'], # which best vars to write
				verbose = 1, # handles output (level 1 is set level, level 2 is episode level)
				on_evaluate = True, # toggle to run modifier on evaluation environ
				on_train = True, # toggle to run modifier on train environ
				frequency = 1, # use modifiation after how many calls to parent method?
				counter = 0, # keepts track of number of calls to parent method
				activate_on_first = True, # will activate on first call otherwise only if % is not 0
				): 
		self.connect_priority = -1 # needs other components to connect first
		self.amp_up_static = np.array(amp_up_static, dtype=float)
		if self.success < 0:
			self.success = self.nEpisodes
		# ger epsilon (value reward must improve by)
		self._epsilon = 1 * 10**(-1*epsilon_order)

	def connect(self, state=None):
		super().connect(state)
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._name + '/'
			if not os.path.exists(self.write_folder):
				os.makedirs(self.write_folder)

	def activate(self, state=None):
		if self.check_counter(state):
			# TEMPORARY CODE TO CHECK IS TRAINING
			# print first weight (just to check if training)
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
		self.best_score = 0
		self.best_eval = 0
		self.set_counter = 0
		self.wait = 0
		self.amping_phase = True

	# steps through one evaluation episode
	def evaluate_episode(self):
		# reset environment, returning first observation
		observation_data = self._evaluate_environment.reset()
		# start episode
		done = False
		while(not done):
			# get rl output
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
		# call end for modifiers
		self._model.end()
		# end of episode
		return state['termination_result'] == 'success', reward

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
			utils.speak(f'Evaluation #{self.set_counter} evaluated with total_success:{total_success} and mean_reward:{mean_reward}')

		# save every model
		self._model.save_model(self.write_folder + 'model_' + str(self.set_counter) + '.zip')

		another_set = False # used to determine if a nother set should run	
		if all_success:
			new_best = False # measures if we improved from last epochs

			# check if we need to switch from amp up to reward optimization
			if self.amping_phase:
				stop_amp_check = self._evaluate_environment._goal.random_dim_min
				if stop_amp_check >= self.stop_amp_at:
					self.best_score = -999999
					self.amping_phase = False
					if self.verbose > 0:
						utils.speak(f'Switching eval phase from amp to reward...')

			# are we optimizing goal distance?
			if self.amping_phase:
				this_score = self._evaluate_environment._goal.random_dim_min
				# amp up goal distance
				self._evaluate_environment._goal.amp_up(self.amp_up_static, self.amp_up_random, self.amp_up_random)
				another_set = True # evaluate again, to see if we can get even farther without more training
				new_best = True # as long as we amped up we are improving

			# are we optimizing reward?
			else:
				this_score = mean_reward
				another_set = False # no need to evaluate again if optimizing reward (should get same reward)
				#new_best = this_score - self.best_score > self._epsilon # did we improve reward?
				new_best = True
				# round for cleaner file output
				this_score = round(this_score, self.epsilon_order)

				

			# do we update best epoch?
			if new_best:
				# update best
				self.best_score = this_score
				self.best_counter = self.counter
				# save best
				if self.amping_phase and 'model' in self.track_vars:
					self._model.save_model(self.write_folder + 'best_model_' + str(this_score) + '.zip')
				if 'replay_buffer' in self.track_vars:
					self._model.save_replay_buffer(self.write_folder + 'best_replay_buffer.zip')
				# update early stopping
				self.wait = 0
				if self.verbose > 0:
					phase_name = 'amp_up' if self.amping_phase else 'reward'
					utils.speak(f'New best score of {this_score}, in phase {phase_name}!!!')
		else:
			# update early stopping
			self.wait += 1
			
		self.set_counter += 1
		return self.wait >= self.patience, another_set
