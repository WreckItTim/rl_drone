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
			  nEpisodes = 1,
			  _write_folder = None, 
			  set_counter = 0,
			  stopping_patience = 4,
			  stopping_percent_success = 0.9,
			  stopping_success_streak = 0,
			  ): 
		# set folder path to write evaluations to
		if _write_folder is None:
			self._write_folder = utils.get_global_parameter('working_directory') + 'evaluations/'
		# create write directory if does not exist already
		if not os.path.exists(self._write_folder):
			os.makedirs(self._write_folder)

	# if reset learning loop
	def reset_stopping(self):
		self.stopping_success_streak = 0
		self.set_counter = 0

	# steps through one evaluation episode
	def evaluate_episode(self):
		# make states object to fill step by step
		states = {}
		# reset environment
		self._evaluate_environment.reset()
		# get first observation
		observation_data, observation_name = self._evaluate_environment._observer.observe()
		# start of episode
		success = False
		for step in range(1, 1_000_000):
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
				if frozen_state['termination_result'] == 'success':
					success = True
				break
		# end of episode
		return states, self._evaluate_environment._nSteps, success

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# keep track of stopping stats
		total_steps = 0
		nSuccess = 0
		# allocate space to save states for all episodes
		all_states = {}
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			states, steps, success = self.evaluate_episode()
			# log results
			all_states['episode_' + str(episode)] = states
			total_steps += steps
			nSuccess += success
		all_states['timestamp'] = utils.get_timestamp()
		# write states to file
		utils.write_json(all_states, self._write_folder + str(self.set_counter) + '.json')
		# counter++
		self.set_counter += 1

		# CHECK STOPPING CRITERIA
		stop = False
		# check percent of evaluations that were successful
		percent_success = nSuccess / self.nEpisodes
		print('Evaluated with success:', percent_success)
		if percent_success >= self.stopping_percent_success:
			self.stopping_success_streak += 1
		else:
			self.stopping_success_streak = 0
		if self.stopping_success_streak >= self.stopping_patience:
			print('Early Stopping Triggered!')
			stop = True
		return stop

	# handle resets while training		
	def reset(self):
		# check when to do next set of evaluations
		if self._train_environment.episode_counter % self.frequency == 0:
			# evaluate for a set of episodes
			stop = self.evaluate_set()
			if stop:
				Configuration.get_active().controller.stop()

	# when using the debug controller
	def debug(self):
		# evaluate for a set of episodes
		stop = self.evaluate_set()
		print('Stopping Criteria Met?', stop)
