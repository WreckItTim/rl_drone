from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import utils
import os

# environment that is heart of drone_rl - for training rl algos
class DroneRL(Environment):
	# even though we do not render, this field is necesary for sb3
	metadata = {"render.modes": ["rgb_array"]}

	# pass in nEpisodes=x if loading from a previous run (continuing training)
	@_init_wrapper
	def __init__(self, 
				 drone_component, 
				 actor_component, 
				 observer_component, 
				 rewarder_component, 
				 terminators_components,
				 spawner_component=None,
				 goal_component=None,
				 evaluator_component=None,
				 saver_component=None,
				 others_components=None,
				 episode_counter=0, 
				 step_counter=0, 
				 is_evaluation_environment=False,
				 ):
		super().__init__()
		self._last_observation_name = 'None'
		self._all_states = {}
		self._observations = {}
		self._last_episode = 1

	def connect(self):
		super().connect()
	
	def clean_rl_output(self, rl_output):
		if np.issubdtype(rl_output.dtype, np.integer):
			return int(rl_output)
		if np.issubdtype(rl_output.dtype, np.floating):
			return rl_output.astype(float).tolist()

	def dump(self, write_folder):
		part_name = 'episodes_' + str(self._last_episode) + '_' + str(self.episode_counter)
		if 'states' in self._dumps:
			path = write_folder + 'states__' + part_name + '.json'
			utils.write_json(self._all_states, path)
			self._all_states = {}
		if 'observations' in self._dumps:
			path = write_folder + 'observations__' + part_name + '.npz'
			np.savez(path, **self._observations)
			self._observations = {}
		self._last_episode = self.episode_counter + 1

	# activate needed components
	def step(self, rl_output):
		# initialize state with rl_output
		rl_output = self.clean_rl_output(rl_output)
		state = {'rl_output':rl_output}
		# increment number of steps
		self._nSteps += 1 # steps this episode
		self.step_counter += 1 # total number of steps
		state['nSteps'] = self._nSteps 
		# take action
		transcribed_action = self._actor.act(rl_output)
		state['transcribed_action'] = transcribed_action
		# get observation
		state['observation_component'] = self._last_observation_name
		observation_data, observation_name = self._observer.observe()
		self._last_observation_name = observation_name
		# set state kinematics variables
		state['drone_position'] = self._drone.get_position()
		state['yaw'] = self._drone.get_yaw() 
		# take step for other components
		if self._others is not None:
			for other in self._others:
				other.step(state)
		# assign rewards (stores total rewards and individual rewards in state)
		total_reward = self._rewarder.reward(state)
		# check for termination
		done = False
		for terminator in self._terminators:
			done = done or terminator.terminate(state)
		state['done'] = done
		# save observation?
		if 'observations' in self._dumps:
			self._observations[observation_name] = observation_data
		# save state?
		if 'states' in self._dumps:
			self._states['step_' + str(self._nSteps)] = state.copy()
			if done: 
				self._all_states['episode_' + str(self.episode_counter)] = self._states.copy()
		if done: 
			self.episode_counter += 1
			# reset savers at end of episodes not begin
			if self._saver is not None:
				self._saver.reset()
		# state is passed to stable-baselines3 callbacks
		return observation_data, total_reward, done, state

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def reset(self):
		# reset all components, several reset() methods may be blank
		# order may matter here, currently no priority queue set-up, may need later
		if self._evaluator is not None:
			self._evaluator.reset()
		self._drone.reset()
		if self._spawner is not None:
			self._spawner.reset()
		if self._others is not None:
			for other in self._others:
				other.reset()
		if self._goal is not None:
			self._goal.reset(self.is_evaluation_environment)
		self._actor.reset()
		self._observer.reset()
		self._rewarder.reset()
		for terminator in self._terminators:
			terminator.reset()

		# init variables
		self._nSteps = 0
		observation_data, observation_name = self._observer.observe()
		self._last_observation_name = observation_name
		state = {'nSteps':self._nSteps}
		state['drone_position'] = self._drone.get_position()
		state['yaw'] = self._drone.get_yaw() 
		state['goal_position'] = self._goal.get_position()

		# track long term vars
		if 'states' in self._dumps:
			self._states = {}
			self._states['step_' + str(self._nSteps)] = state.copy()

		return observation_data