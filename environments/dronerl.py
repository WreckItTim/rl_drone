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
				 evaluator_component=None,
				 saver_component=None,
				 others_components=None,
				 write_observations=False,
				 episode_counter=0, 
				 step_counter=0, 
				 ):
		super().__init__()

	def connect(self):
		super().connect()
			
	# activate needed components
	def step(self, rl_output):
		# initialize state with rl_output
		state = {'rl_output':float(rl_output)}
		# increment number of steps
		self._nSteps += 1
		self.step_counter += 1
		state['nSteps'] = self._nSteps 
		# take action
		transcribed_action = self._actor.act(rl_output)
		state['transcribed_action'] = transcribed_action
		# get observation
		observation_data, observation_name = self._observer.observe(self.write_observations)
		state['observation_component'] = observation_name
		# set state kinematics variables
		state['drone_position'] = self._drone.get_position()
		state['yaw'] = self._drone.get_yaw() 
		# take step for other components
		if self._others is not None:
			for other in self._others:
				other.step(state)
		if 'goal' in state and self._nSteps == 1:
			goal_yaw = utils.position_to_yaw(state['goal'])
			state['goal_yaw'] = goal_yaw
			print('goal:', state['goal'], goal_yaw)
		# assign rewards (stores total rewards and individual rewards in state)
		total_reward = self._rewarder.reward(state)
		# check for termination
		done = False
		for terminator in self._terminators:
			done = done or terminator.terminate(state)
		state['done'] = done
		if done:
			print('terminated:', state['termination_reason'], state['termination_result'])
			self.episode_counter += 1
		# state is passed to stable-baselines3 callbacks
		return observation_data, total_reward, done, state

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def reset(self):
		if self.write_observations:
			print('eval eps', self.episode_counter)
		else:
			print('train eps', self.episode_counter)
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
		if self._others is not None:
			for other in self._others:
				other.reset()
		self._actor.reset()
		self._observer.reset()
		self._rewarder.reset()
		for terminator in self._terminators:
			terminator.reset()
		self._nSteps = 0
		observation_data, observation_name = self._observer.observe()
		print('spawn:', self._drone.get_position(), self._drone.get_yaw(),)
		return observation_data