from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import utils

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
				 episode_counter=0, 
				 step_counter=0, 
				 is_evaluation_environment=False,
				 ):
		super().__init__()
		self._last_observation_name = 'None'

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
		state = {'rl_output':rl_output}
		# increment number of steps
		self._nSteps += 1
		self.step_counter += 1
		state['nSteps'] = self._nSteps 
		# take action
		transcribed_action = self._actor.act(rl_output)
		state['transcribed_action'] = transcribed_action
		# get observation
		state['observation_component'] = self._last_observation_name
		observation_data, observation_name = self._observer.observe(self.write_observations)
		self._last_observation_name = observation_name
		# set state kinematics variables
		state['drone_position'] = self._drone.get_position()
		state['yaw'] = self._drone.get_yaw() 
		state['goal_position'] = self._goal.get_position()
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
		#prefix = 'evaluate' if self.is_evaluation_environment else 'train'
		#utils.write_json(state, 'temp/states/' + prefix + '_episode' + str(self.episode_counter) + '_step' + str(self._nSteps) + '.json')
		if done:
			self.episode_counter += 1
		#x = input()
		# state is passed to stable-baselines3 callbacks
		return observation_data, total_reward, done, state

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def reset(self):
		#print(self.is_evaluation_environment, self.episode_counter, self._drone.get_state())
		#if self.write_observations:
		#	print('evaluation episode', self.episode_counter)
		#else:
		#	print('train episode', self.episode_counter)
		# reset all components, several reset() methods may be blank
		# order may matter here, currently no priority queue set-up, may need later
		if self._saver is not None:
			self._saver.reset()
		if self._evaluator is not None:
			stop = self._evaluator.reset()
			if stop:
				raise Exception('EARLY STOPPING TRIGGERED, learning complete')
		self._drone.reset()
		if self._spawner is not None:
			self._spawner.reset()
		#print()
		#print('spawn:', utils._round(self._drone.get_position()), utils._round(self._drone.get_yaw(),))
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
		
		state = {'nSteps':0}
		state['drone_position'] = self._drone.get_position()
		state['yaw'] = self._drone.get_yaw() 
		state['goal_position'] = self._goal.get_position()
		prefix = 'evaluate' if self.is_evaluation_environment else 'train'
		print('reset', prefix, self.episode_counter)
		#utils.write_json(state, 'temp/states/' + prefix + '_episode' + str(self.episode_counter) + '_step0.json')
		#print(self.is_evaluation_environment, self.episode_counter, self._drone.get_state())
		return observation_data