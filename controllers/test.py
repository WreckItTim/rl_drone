from controllers.controller import Controller
from component import _init_wrapper
import rl_utils as utils
import numpy as np
import torch

# simply runs an evaluation set on the given evaluation environment
class Test(Controller):
	# constructor
	@_init_wrapper
	def __init__(self,
				environment_component, # environment to run eval in
				model_component, # used to make predictions
				results_directory,
				num_episodes,
				 ):
		super().__init__()

	# runs control on components
	def run(self):
		self._configuration.reset_all()
		self.evaluate_set()

	def connect(self):
		super().connect()

	# steps through one evaluation episode
	def evaluate_episode(self):
		# reset environment, returning first observation
		observation_data, state = self._environment.reset()
		# start episode
		done = False
		rewards = []
		gamma = 0.99
		while(not done):
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, truncated, state = self._environment.step(rl_output)
			rewards.append(reward)
		q = 0
		for i, reward in enumerate(rewards):
			q += reward * gamma**(len(rewards)-i-1)
		# end of episode
		return state['reached_goal'], state['termination_reason'], q

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# loop through all episodes
		successes = []
		termination_reasons = []
		qs = []
		for episode in range(self.num_episodes):
			# step through next episode
			success, termination_reason, q = self.evaluate_episode()
			utils.speak(f'episode:{episode} goal:{success} q:{q} termination:{termination_reason}')
			successes.append(success)
			termination_reasons.append(termination_reason)
			qs.append(q)
		results_dic = {
			'successes':successes,
			'termination_reasons':termination_reasons,
			'qs':qs,
		}
		results_path = self.results_directory + 'evaluation.json'
		utils.write_json(results_dic, results_path)

