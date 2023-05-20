from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import rl_utils as utils
import os

# an environment is the heart of RL algorithms
# the Goal flavor wants the drone to go to Point A to Point B
# steps until termination then resets
# if a saver modifier is used, will write observations and states
class GoalEnv(Environment):

	# constructor
	# if continuing training, step_counter and episode_counter will likely be > 0
	@_init_wrapper
	def __init__(self, 
				 drone_component, 
				 actor_component, 
				 observer_component, 
				 rewarder_component, 
				 goal_component,
				 model_component,
				 others_components=None,
				 step_counter=0, 
				 episode_counter=0, 
				 save_counter=0,
				 is_evaluation_env=False,
				 ):
		super().__init__()
		self._last_observation_name = 'None'
		self._all_states = {}
		self._observations = {}
		self._track_save = False
		
	# this will toggle if keep track of observations and states
	# note this is expensive, so must dump using save() from time to time
	# best route to do this is a saver modifier
	# track_vars are strings with which variables to save
	def set_save(self,
			  track_save,
			  track_vars=[
				  'observations', 
				  'states',
				  ],
			  ):
		self._track_save = track_save
		self._track_vars = track_vars.copy()

	# if reset learning loop
	def reset_learning(self):
		self.step_counter = 0 # total steps
		self.episode_counter = 0
		self.save_counter = 0
		self._all_states = {}
		self._observations = {}

	# save observations and states
	# these are done in chunks rather than individual files
		# this saves memory, reduces zip time, and avoids clutter
	# pass in write_folder to state
	def save(self, state):
		write_folder = state['write_folder']
		if not self._track_save:
			utils.warning('called GoalEnv.save() without setting _track_save=True, nothing to save')
			return
		part_name = 'part_' + str(self.save_counter)
		if 'states' in self._track_vars:
			path = write_folder + 'states__' + part_name + '.json'
			utils.write_json(self._all_states, path)
			self._all_states = {}
		if 'observations' in self._track_vars:
			path = write_folder + 'observations__' + part_name + '.npz'
			np.savez(path, **self._observations)
			self._observations = {}
		self.save_counter += 1

	# activate needed components
	def step(self, rl_output, state=None):
		# next step
		self._nSteps += 1 # total number of steps
		self.step_counter += 1 # total number of steps
		this_step = 'step_' + str(self._nSteps)
		if state is None:
			self._states[this_step] = {}
		else:
			self._states[this_step] = state.copy()
		self._states[this_step]['nSteps'] = self._nSteps
		self._states[this_step]['is_evaluation_env'] = self.is_evaluation_env
		# clean and save rl_output to state
		self._states[this_step]['rl_output'] = list(rl_output)
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
		# also checks if we are done with episode
		total_reward, done = self._rewarder.step(self._states[this_step])
		self._states[this_step]['done'] = done
		# save data?
		if self._track_save and 'observations' in self._track_vars:
			self._observations[observation_name] = observation_data.copy()
		if self._track_save and 'states' in self._track_vars and done: 
			self._all_states['episode_' + str(self.episode_counter)] = self._states.copy()
		if done: 
			self.end(self._states[this_step])
		# data needed to relay states to replay buffer and state
		return observation_data, total_reward, done, self._states[this_step]

	# called at beginning of each episode to prepare for next
	# returns first observation for new episode
	# spawn_to will overwrite previous spawns and force spawn at that x,y,z,yaw
	def start(self, state = None):
		self.episode_counter += 1
		# init state(s)
		self._nSteps = 0 # steps this episode
		this_step = 'step_' + str(self._nSteps)
		if state is None:
			self._states = {this_step:{}}
		else:
			self._states = {this_step:state.copy()}
		self._states[this_step]['nSteps'] = self._nSteps
		self._states[this_step]['is_evaluation_env'] = self.is_evaluation_env

		# reset drone and goal components, several start() methods may be blank
		# order may matter here, currently no priority queue set-up, may need later
		self._model.start(self._states[this_step])
		self._drone.start(self._states[this_step])
		if state is not None and 'spawn_to' in state:
			self._drone.teleport(*state['spawn_to'], ignore_collision=True)
		self._states[this_step]['drone_position'] = self._drone.get_position()
		self._states[this_step]['yaw'] = self._drone.get_yaw() 

		self._goal.start(self._states[this_step])
		if state is not None and 'goal_at' in state:
			self._goal.set_position(*state['goal_at'][:3])
		self._states[this_step]['goal_position'] = self._goal.get_position()

		# start other components
		if self._others is not None:
			for other in self._others:
				other.start(self._states[this_step])
		self._actor.start(self._states[this_step])
		self._observer.start(self._states[this_step])
		self._rewarder.start(self._states[this_step])

		# get first observation
		observation_data, observation_name = self._observer.step(self._states[this_step])
		self._last_observation_name = observation_name
		
		# save data?
		if self._track_save and 'observations' in self._track_vars:
			self._observations[observation_name] = observation_data.copy()

		return observation_data

	# called at the end of each episode for any clean up, when done=True
	# normally only start() is used in OpenAI Gym environments
	# but end() is much easier/clear/necessary for modifiers and multiple envs
	# off-by-one errors aggregate when switching between multiple envs
	def end(self, state=None):
		# end all components
		self._model.end(state)
		self._drone.end(state)
		self._goal.end(state)
		if self._others is not None:
			for other in self._others:
				other.end(state)
		self._actor.end(state)
		self._observer.end(state)
		self._rewarder.end(state)