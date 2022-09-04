from others.other import Other
from component import _init_wrapper
import utils
import os

# objective is set x-meters in front of drone and told to go forward to it
class Evaluator(Other):
	@_init_wrapper
	def __init__(self, 
			  environment_component,
			  model_component,
			  frequency = 1000,
			  nEpisodes = 1,
			  _write_folder = None, 
			  set_counter = 0,
			  ): 
		# set folder path to write evaluations to
		if _write_folder is None:
			self._write_folder = utils.get_global_parameter('working_directory') + 'evaluations/'
		# create write directory if does not exist already
		if not os.path.exists(self._write_folder):
			os.makedirs(self._write_folder)

	# steps through one evaluation episode
	def evaluate_episode(self):
		# make states object to fill step by step
		states = {}
		# reset environment
		self._environment.reset()
		# get first observation
		observation_numpy = self._environment._observer.observe().to_numpy()
		# start of episode
		for step in range(1, 1_000_000):
			# get rl output
			rl_output = self._model.predict(observation_numpy)
			# take next step
			observation_numpy, reward, done, state = self._environment.step(rl_output)
			# freeze state
			frozen_state = state.copy()
			# save state
			states[step] = frozen_state
			# check if we are done
			if done:
				break
		# end of episode
		return states

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# allocate space to save states for all episodes
		all_states = {}
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			all_states[episode] = self.evaluate_episode()
		# write states to file
		utils.write_json(all_states, self._write_folder + str(self.set_counter) + '.json')
		# counter++
		self.set_counter += 1

	# handle resets while training		
	def reset(self):
		# check when to do next set of evaluations
		if self._environment.episode_counter % self.frequency == 0:
			# evaluate for a set of episodes
			self.evaluate_set()

	# when using the debug controller
	def debug(self):
		# evaluate for a set of episodes
		self.evaluate_set()