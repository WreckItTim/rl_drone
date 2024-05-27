from controllers.controller import Controller
from component import _init_wrapper
import rl_utils as utils
import os
import torch
import numpy as np

# simply runs an evaluation set on the given evaluation environment
class Evaluate(Controller):
	# constructor
	@_init_wrapper
	def __init__(self,
				evaluate_environment_component, # environment to run eval in
				model_path, # used to make predictions, track best model
				nEpisodes,
				 ):
		super().__init__()

	# runs control on components
	def run(self):
		self.evaluate_set()

	def connect(self):
		super().connect()
		self.write_folder = utils.get_global_parameter('working_directory')
		self.write_folder += 'Evaluate/'
		if not os.path.exists(self.write_folder):
			os.makedirs(self.write_folder)
		from stable_baselines3 import TD3 as sb3TD3
		import stable_baselines3 as sb3
		print(self.model_path, sb3.__version__)
		self._model = sb3TD3.load(self.model_path)

	# steps through one evaluation episode
	def evaluate_episode(self):
		# reset environment, returning first observation
		observation_data = self._evaluate_environment.reset()[0]
		# start episode
		done = False
		rewards = []
		gamma = 0.99
		while(not done):
			# get rl output
			observation_data_np = np.expand_dims(observation_data, axis=0)
			observation_data_th = torch.from_numpy(observation_data_np).to('cuda')
			rl_output = self._model.actor(observation_data_th)[0].detach().cpu().numpy()
			# take next step
			observation_data, reward, done, truncated, state = self._evaluate_environment.step(rl_output)
			rewards.append(reward)
		total_reward = 0
		for i, reward in enumerate(rewards):
			total_reward += reward * gamma**(len(rewards)-i-1)
		# end of episode
		return state['termination_result'] == 'success', total_reward

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# loop through all episodes
		for episode in range(self.nEpisodes):
			# step through next episode
			this_success, this_reward = self.evaluate_episode()
			utils.speak(f'evaluated episode {episode} with reward {this_reward} and success {this_success}')
		# state = {'write_folder':'local/runs/eval2_V1/EvaluateEnvironment'}
		# self._evaluate_environment.set_save(
		# 	  track_save=True,
		# 	  track_vars=[
		# 		  'observations', 
		# 		  'states',
		# 		  ],)
		# self._evaluate_environment.save(state)
