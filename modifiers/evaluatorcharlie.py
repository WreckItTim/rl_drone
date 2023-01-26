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
			  amp_up_static = [4, 0, 0], # increases static goal distance
			  amp_up_random = 4, # increases random goal distance
			  nEpisodes = 1, # number of episodes to evaluate each set
			  success = -1, # number of successfull episodes for set success, -1=all
			  patience = 100, # number of sets to wait to improve best_score
			  wait = 0, # number of sets have been waiting to improve score
			  set_counter = 0, # count number of eval sets
			  random = False, # set true to select randomly from spawn objects
			  write_folder = None, # writes best model / replay buffer here
			  track_vars = ['model', 'replay_buffer'], # which best vars to write
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

	def connect(self, state=None):
		super().connect(state)
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._name + '/'
			if not os.path.exists(self.write_folder):
				os.makedirs(self.write_folder)

	def activate(self, state=None):
		if self.check_counter(state):
			# evaluate for a set of episodes, until failure
			while(True):
				stop, total_success = self.evaluate_set()
				if not total_success:
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
		# end of episode
		return state['termination_result'] == 'success'

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# keep track of episode results
		nSuccess = 0
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			nSuccess += self.evaluate_episode()

		if self.verbose > 0:
			utils.speak(f'Evaluation #{self.set_counter} evaluated with nSuccess:{nSuccess}')
			
		total_success = nSuccess >= self.success
		if total_success:
			# amp up goal distance
			self._evaluate_environment._goal.amp_up(self.amp_up_static, self.amp_up_random, self.amp_up_random)
			# update best
			self.best_score = self._evaluate_environment._goal.random_dim_min
			self.best_eval = self.set_counter
			# save best
			if 'model' in self.track_vars:
				self._model.save_model(self.write_folder + 'best_model.zip')
			if 'replay_buffer' in self.track_vars:
				self._model.save_replay_buffer(self.write_folder + 'best_replay_buffer.zip')
			# update early stopping
			self.wait = 0
			if self.verbose > 0:
				utils.speak(f'Amped up goal distance to {self.best_score}')
		else:
			# update early stopping
			self.wait += 1

		self.set_counter += 1
		return self.wait >= self.patience, total_success
