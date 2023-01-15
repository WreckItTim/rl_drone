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
			  best_eval = 0,
			  patience = 64,
			  wait = 0,
			  curriculum = True,
			  goal_component = None,
			  steps_components = None,
			  ): 
		self.connect_priority = -1 # needs other components to connect first

	# get initial learning loop values for curriculum
	def connect(self):
		super().connect()
		if self.curriculum:
			self._goal_xyz_point = self._goal.xyz_point.copy()
			self._goal_random_dim_min = self._goal.random_dim_min
			self._goal_random_dim_max = self._goal.random_dim_max
			self._step_max_steps = [0] * len(self._steps)
			for idx, step in enumerate(self._steps):
				self._step_max_steps[idx] = step.max_steps

	# if reset learning loop
	def reset_learning(self):
		self.best = 0
		self.evaluation_counter = 0
		if self.curriculum:
			self._goal.xyz_point = self._goal_xyz_point.copy()
			self._goal.random_dim_min = self._goal_random_dim_min
			self._goal.random_dim_max = self._goal_random_dim_max
			for idx, step in enumerate(self._steps):
				step.max_steps = self._step_max_steps[idx]

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

		mean_reward = total_reward / self.nEpisodes
		print('Evaluated with average reward:', mean_reward)

		stop = False
		success = False
		if mean_reward > self.stopping_reward:
			success = True
			self._goal.xyz_point += np.array([4, 0, 0], dtype=float)
			self._goal.random_dim_min += 4
			self._goal.random_dim_max += 4
			self.best = self._goal.random_dim_min
			self._model._best_goal = self.best # relay information to hyper search
			self.best_eval = self.evaluation_counter
			for step in self._steps:
				step.max_steps = 8 + self._goal.random_dim_max
			self.wait = 0
			print('Amping up distance to goal to', self._goal.random_dim_min)
		else:
			self.wait += 1
		if self.wait >= self.patience:
			stop = True

		self.evaluation_counter += 1
		return stop, success

	# handle resets while training		
	def reset(self):
		# check when to do next set of evaluations
		if self._train_environment.episode_counter % self.frequency == 0:
			# evaluate for a set of episodes, until failure
			while True:
				stop, success = self.evaluate_set()
				if not success:
					break
			# close up shop
			if stop:
				Configuration.get_active().controller.stop()

	# when using the debug controller
	def debug(self):
		# evaluate for a set of episodes
		stop = self.evaluate_set()
		print('Stopping Criteria Met?', stop)
