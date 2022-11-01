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
				 write_observations=False,
				 write_states=False,
				 episode_counter=0, 
				 step_counter=0, 
				 is_evaluation_environment=False,
				 directory_path = None,
				 ):
		super().__init__()
		self._last_observation_name = 'None'
		if directory_path is None:
			self.directory_path = utils.get_global_parameter('working_directory') + '/states/'
		if not os.path.exists(self.directory_path):
			os.mkdir(self.directory_path)

	def connect(self):
		super().connect()
	
	def clean_rl_output(self, rl_output):
		if np.issubdtype(rl_output.dtype, np.integer):
			return int(rl_output)
		if np.issubdtype(rl_output.dtype, np.floating):
			return rl_output.astype(float).tolist()

	# activate needed components
	def step(self, rl_output):
		# initialize state with rl_output
		rl_output = self.clean_rl_output(rl_output)
		self._state = {'rl_output':rl_output}
		# increment number of steps
		self._nSteps += 1
		self.step_counter += 1
		self._state['nSteps'] = self._nSteps 
		# take action
		transcribed_action = self._actor.act(rl_output)
		self._state['transcribed_action'] = transcribed_action
		# get observation
		self._state['observation_component'] = self._last_observation_name
		observation_data, observation_name = self._observer.observe(self.write_observations)
		self._last_observation_name = observation_name
		# set state kinematics variables
		self._state['drone_position'] = self._drone.get_position()
		self._state['yaw'] = self._drone.get_yaw() 
		self._state['goal_position'] = self._goal.get_position()
		# take step for other components
		if self._others is not None:
			for other in self._others:
				other.step(self._state)
		# assign rewards (stores total rewards and individual rewards in state)
		total_reward = self._rewarder.reward(self._state)
		# check for termination
		done = False
		for terminator in self._terminators:
			done = done or terminator.terminate(self._state)
		self._state['done'] = done
		self._state_space['step_' + str(self._nSteps)] = self._state.copy()
		if done:
			if self.write_states:
				utils.write_json(self._state_space, self.directory_path + 'episode_' + str(self.episode_counter) + '.json')
			self.episode_counter += 1
		# state is passed to stable-baselines3 callbacks
		return observation_data, total_reward, done, self._state

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def reset(self):
		# reset all components, several reset() methods may be blank
		# order may matter here, currently no priority queue set-up, may need later
		if self._saver is not None:
			self._saver.reset()
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
		self._nSteps = 0
		observation_data, observation_name = self._observer.observe()
		self._last_observation_name = observation_name
		
		self._state = {'nSteps':self._nSteps}
		self._state['drone_position'] = self._drone.get_position()
		self._state['yaw'] = self._drone.get_yaw() 
		self._state['goal_position'] = self._goal.get_position()
		self._state_space = {'step_0':self._state.copy()}

		return observation_data