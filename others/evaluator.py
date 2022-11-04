from others.other import Other
from component import _init_wrapper
import utils
import os
from configuration import Configuration
import numpy as np

# objective is set x-meters in front of drone and told to go forward to it
class Evaluator(Other):
	@_init_wrapper
	def __init__(self, 
			  train_environment_component,
			  evaluate_environment_component,
			  model_component,
			  frequency = 1000,
			  nEpisodes = 100,
			  evaluation_counter = 0,
			  stopping_reward = 9,
			  best = 0,
			  write_best_model_path = None,
			  curriculum = True,
			  goal_component=None,
			  steps_components=None,

			  ): 
		# set where to save model
		if write_best_model_path is None:
			self.write_best_model_path = utils.get_global_parameter('working_directory') + 'best_model'
		# keep track of num evaluations regardless of continuing training or not
		self._this_counter = 0

	# if reset learning loop
	def reset_stopping(self):
		self.best = 0
		self.evaluation_counter = 0

	# steps through one evaluation episode
	def evaluate_episode(self):
		# reset environment, returning first observation
		observation_data = self._evaluate_environment.reset()
		# start episode
		done = False
		while(True):
			# get rl output
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
			# check if we are done
			if done:
				break
		# end of episode
		return reward

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# keep track of stopping stats
		total_reward = 0
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			reward = self.evaluate_episode()
			# log results
			total_reward += reward
		# counter++
		self.evaluation_counter += 1
		self._this_counter += 1

		# CHECK STOPPING CRITERIA and best model
		stop = False
		mean_reward = total_reward / self.nEpisodes
		print('Evaluated with average reward:', mean_reward)
		# check for best model
		if mean_reward > self.best:
			self._model.save(self.write_best_model_path)
			self.best = mean_reward
		# check stopping criteria
		if mean_reward > self.stopping_reward:
			print('Stopping criteria met!')
			stop = True
		return stop

	# handle resets while training		
	def reset(self):
		# check when to do next set of evaluations
		if self._train_environment.episode_counter % self.frequency == 0:
			# skip evaluation 0 if continuing training
			if self._this_counter == 0 and self.evaluation_counter > 0:
				self._this_counter += 1
			else:
				# evaluate for a set of episodes
				stop = self.evaluate_set()
				if stop:
					if self.curriculum and self._goal.random_dim_max <= 100:
						self._goal.xyz_point += np.array([4, 0, 0], dtype=float)
						self._goal.random_dim_min += 4
						self._goal.random_dim_max += 4
						for step in self._steps:
							step.max_steps = 8 + self._goal.random_dim_max
						print('Amping up distance to goal to', self._goal.random_dim_min)
					else:
						Configuration.get_active().controller.stop()

	# when using the debug controller
	def debug(self):
		# evaluate for a set of episodes
		stop = self.evaluate_set()
		print('Stopping Criteria Met?', stop)
