from others.other import Other
from component import _init_wrapper
import utils
import os
from configuration import Configuration

# objective is set x-meters in front of drone and told to go forward to it
class Evaluator(Other):
	@_init_wrapper
	def __init__(self, 
			  train_environment_component,
			  evaluate_environment_component,
			  model_component,
			  frequency = 1000,
			  nEpisodes = 100,
			  write_evaluations_folder = None, 
			  evaluation_counter = 0,
			  stopping_patience = 4,
			  stopping_reward = 9,
			  stopping_best = 0,
			  stopping_wait = 0,
			  stopping_epsilon = 1e-4,
			  save_best_model = True,
			  write_best_model_path = None,
			  curriculum = True,
			  goal_component=None,
			  steps_components=None,

			  ): 
		# set folder path to write evaluations to
		if write_evaluations_folder is None:
			self.write_evaluations_folder = utils.get_global_parameter('working_directory') + 'evaluations/'
		# create write directory if does not exist already
		if not os.path.exists(self.write_evaluations_folder):
			os.makedirs(self.write_evaluations_folder)
		# set where to save model
		if write_best_model_path is None:
			self.write_best_model_path = utils.get_global_parameter('working_directory') + 'best_model'
		# keep track of num evaluations regardless of continuing training or not
		self._this_counter = 0

	# if reset learning loop
	def reset_stopping(self):
		self.stopping_best = 0
		self.stopping_wait = 0
		self.evaluation_counter = 0

	# steps through one evaluation episode
	def evaluate_episode(self):
		# make states object to fill step by step
		states = {}
		# reset environment, returning first observation
		observation_data = self._evaluate_environment.reset()
		# start episode
		done = False
		for step in range(1, 10_000):
			# get rl output
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
			# freeze state
			frozen_state = state.copy()
			# save state
			states['step_' + str(step)] = frozen_state
			# check if we are done
			if done:
				break
		# end of episode
		return states, reward

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# keep track of stopping stats
		total_reward = 0
		# allocate space to save states for all episodes
		all_states = {}
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			states, reward = self.evaluate_episode()
			# log results
			all_states['episode_' + str(episode)] = states
			total_reward += reward
		all_states['timestamp'] = utils.get_timestamp()
		# write states to file
		utils.write_json(all_states, self.write_evaluations_folder + 'evaluation_' + str(self.evaluation_counter) + '.json')
		# counter++
		self.evaluation_counter += 1
		self._this_counter += 1

		# CHECK STOPPING CRITERIA
		stop = False
		mean_reward = total_reward / self.nEpisodes
		print('Evaluated with average reward:', mean_reward)
		if mean_reward - self.stopping_best >= self.stopping_epsilon:
			# save best model
			if self.save_best_model:
				self._model.save(self.write_best_model_path)
			self.stopping_wait = 0
			self.stopping_best = mean_reward
		else:
			self.stopping_wait += 1
		if self.stopping_best >= self.stopping_reward and self.stopping_wait >= self.stopping_patience:
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
					if self.curriculum:
						self.goal.xyz_point += np.array([4, 0, 0], dtype=float)
						self.goal.random_dim_min += 4
						self.goal.random_dim_max += 4
						if self.steps is not None:
							for step in self.steps:
								step.max_steps = self.goal.random_dim_max
						print('Amping up distance to goal to', self.goal.random_dim_min)
					else:
						Configuration.get_active().controller.stop()

	# when using the debug controller
	def debug(self):
		# evaluate for a set of episodes
		stop = self.evaluate_set()
		print('Stopping Criteria Met?', stop)
