from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import utils
import os

# an environment is the heart of RL algorithms
# the Goal flavor wants the drone to go to Point A to Point B
# steps until termination then resets
# if a saver modifier is used, will write observations and states
class GoalEnv(Environment):
	# even though we do not render, this field is necesary for sb3
	metadata = {"render.modes": ["rgb_array"]}

	# constructor
	# if continuing training, step_counter and episode_counter will likely be > 0
	@_init_wrapper
	def __init__(self, 
				 drone_component, 
				 actor_component, 
				 observer_component, 
				 rewarder_component, 
				 terminators_components,
				 goal_component,
				 others_components=None,
				 step_counter=0, 
				 episode_counter=0, 
				 is_evaluation_env=False,
				 ):
		super().__init__()
		self._last_observation_name = 'None'
		self._all_states = {}
		self._observations = {}
		self._last_episode = self.episode_counter
		self._track_save = False
		self._all_states = {}
		self._observations = {}
		
	# this will toggle if keep track of observations and states
	# note this is expensive, so must dump using save() from time to time
	# best route to do this is a saver modifier
	# track_vars are strings with which variables to save
	def set_save(self
			  , track_save
			  , track_vars=[
				  'observations', 
				  'states',
				  ]
			  ):
		self._track_save = track_save
		self._track_vars = track_vars.copy()

	# if reset learning loop
	def reset_learning(self):
		self.step_counter = 0 # total steps
		self.episode_counter = 0
		self._last_episode = self.episode_counter
		self._all_states = {}
		self._observations = {}

	# save observations and states
	# these are done in chunks rather than individual files
		# this saves memory, reduces zip time, and avoids clutter
	def save(self, write_folder):
		if not self._track_save:
			utils.warning('called VanillaEnv.save() without setting _track_save=True, nothing to save')
			return
		part_name = 'episodes_' + str(self._last_episode) + '_' + str(self.episode_counter)
		if 'states' in self._track_vars:
			path = write_folder + 'states__' + part_name + '.json'
			utils.write_json(self._all_states, path)
			self._all_states = {}
		if 'observations' in self._track_vars:
			path = write_folder + 'observations__' + part_name + '.npz'
			np.savez(path, **self._observations)
			self._observations = {}
		self._last_episode = self.episode_counter
	
	# just makes the rl_output from SB3 more readible
	def clean_rl_output(self, rl_output):
		if np.issubdtype(rl_output.dtype, np.integer):
			return int(rl_output)
		if np.issubdtype(rl_output.dtype, np.floating):
			return rl_output.astype(float).tolist()

	# activate needed components
	def step(self, rl_output):
		# next step
		self._nSteps += 0 # steps this episode
		self.step_counter += 1 # total number of steps
		this_step = 'step_'+str(self._nSteps)
		self._states[this_step]['nSteps'] = self._nSteps
		# clean and save rl_output to state
		self._states[this_step]['rl_output'] = self.clean_rl_output(rl_output)
		# take action
		self._actor.step(self._states[this_step])
		# save state kinematics
		self._states[this_step]['drone_position'] = self._drone.get_position()
		self._states[this_step]['yaw'] = self._drone.get_yaw() 
		# get observation
		self._states[this_step]['observation_name'] = self._last_observation_name
		observation_data, observation_name = self._observer.step(self._states[this_step])
		self._last_observation_name = observation_name
		# take step for other components
		if self._others is not None:
			for other in self._others:
				other.step(self._states[this_step])
		# assign rewards (stores total rewards and individual rewards in state)
		total_reward = self._rewarder.step(self._states[this_step])
		# check for termination
		done = False
		for terminator in self._terminators:
			done = done or terminator.terminate(self._states[this_step])
		state['done'] = done
		# save data?
		if self._track_save and 'observations' in self._track_vars:
			self._observations[observation_name] = observation_data.copy()
		if self._track_save and 'states' in self._track_vars and done: 
			self._all_states['episode_' + str(self.episode_counter)] = self._states[this_step].copy()
		if done: 
			self.episode_counter += 1
		# state is passed to stable-baselines3 callbacks
		return observation_data, total_reward, done, None

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def reset(self):
		# reset all components, several reset() methods may be blank
		# order may matter here, currently no priority queue set-up, may need later
		self._drone.reset()
		self._goal.reset()
		if self._others is not None:
			for other in self._others:
				other.reset()
		self._actor.reset()
		self._observer.reset()
		self._rewarder.reset()
		for terminator in self._terminators:
			terminator.reset()

		# init state(s)
		self._nSteps = 0 # steps this episode
		this_step = 'step_'+str(self._nSteps)
		self._states = {this_step:{}}
		self._states[this_step]['drone_position'] = self._drone.get_position()
		self._states[this_step]['yaw'] = self._drone.get_yaw() 
		self._states[this_step]['goal_position'] = self._goal.get_position()

		# get first observation
		observation_data, observation_name = self._observer.step(self._state)
		self._last_observation_name = observation_name
		
		# save data?
		if self._track_save and 'observations' in self._track_vars:
			self._observations[observation_name] = observation_data.copy()

		return observation_data