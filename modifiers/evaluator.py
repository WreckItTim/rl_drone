from modifiers.modifier import Modifier
from component import _init_wrapper
import utils
from configuration import Configuration
import numpy as np

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
			  goal_component, # environment to run eval in
			  best_score = 0, # metric to improve
			  best_eval = 0, # evaluation set corresponding to best_score
			  amp_up_static = [4, 0, 0], # increases static goal distance
			  amp_up_random = 4, # increases random goal distance
			  nEpisodes = 1, # number of episodes to evaluate each set
			  nSuccess = -1, # number of successfull episodes for set success, -1=all
			  patience = 64, # number of sets to wait to improve best_score
			  wait = 0, # number of sets have been waiting to improve score
			  set_counter = 0, # count number of eval sets
			  random = False, # set true to select randomly from spawn objects
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = False, # will activate on first call otherwise only if % is not 0
			  ): 
		self.connect_priority = -1 # needs other components to connect first
		self.amp_up_static = np.array(amp_up_static, dtype=float)
		if self.nSuccess < 0:
			self.nSuccess = self.nEpisodes

	# reset learning loop to static values from connect()
	def reset_learning(self):
		self.best_score = 0
		self.best_eval = 0
		self.evaluation_counter = 0
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
		return state['termination_result']

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# keep track of episode results
		nSuccess = 0
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			nSuccess += self.evaluate_episode()

		if self.verbose > 0:
			utils.speak(f'Evaluation #{self.evaluation_counter} evaluated with nSuccess:{nSuccess}')

		total_success = nSuccess >= self.success
		if total_success:
			# amp up goal distance
			self._goal.xyz_point += self.amp_up_static
			self._goal.random_dim_min += self.amp_up_random
			self._goal.random_dim_max += self.amp_up_random
			# update best
			self.best = self._goal.random_dim_min
			self.best_eval = self.evaluation_counter
			# save best
			self._model.save(self.best_model_path)
			self._model.save_replay_buffer(self.best_replay_buffer_path)
			# update early stopping
			self.wait = 0
			if self.verbose > 1:
				print(f'Amping up goal distance to {self._goal.random_dim_min}')
		else:
			# update early stopping
			self.wait += 1

		self.evaluation_counter += 1
		return self.wait >= self.patience, total_success

	# handle resets while training		
	def reset(self):
		# check when to do next set of evaluations
		if self.reset_counter % self.frequency == 0:
			# evaluate for a set of episodes, until failure
			total_success = False
			while not total_success:
				stop, total_success = self.evaluate_set()
			# close up shop?
			if stop:
				Configuration.get_active().controller.stop()
		self.reset_counter += 1

	# when using the debug controller
	def debug(self):
		# evaluate for a set of episodes
		stop, total_success = self.evaluate_set()
		print('Total Success?', total_success, 'Stopping Criteria Met?', stop)