from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import rl_utils as utils
import os
from stable_baselines3.common.type_aliases import TrainFreq
from gymnasium import spaces
import msgpackrpc

# an environment is the heart of RL algorithms
# the Goal flavor wants the drone to go to Point A to Point B
# Aux adds slimmable methods
# steps until termination then resets
# if a saver modifier is used, will write observations and states
class AuxEnv(Environment):
	# even though we do not render, this field is necesary for sb3
	metadata = {"render.modes": ["rgb_array"]}

	# constructor
	# if continuing training, step_counter and episode_counter will likely be > 0
	@_init_wrapper
	def __init__(self, 
				actor_component, 
				model_component, # aux model for rho-preds
				navi_component, # goalenv environment for navigation
				step_counter=0, 
				episode_counter=0, 
		):
		super().__init__()

	# if reset learning loop
	def reset_learning(self):
		self.step_counter = 0 # total steps
		self.episode_counter = 0

	def connect(self, state=None):
		super().connect()
		# even though we do not directly use the observation or action space, these fields are necesary for sb3
		_output_shape = self._navi._observer._output_shape
		_output_shape = (_output_shape[0] + len(self._navi._actor._actions) + len(self._actor._actions), )
		self.observation_space = spaces.Box(0, 1, shape=_output_shape, dtype=float)
		self.action_space = self._actor.get_space()

	# activate needed components
	def step(self, rl_output, state=None):
		try:
			self.step_counter += 1 # total number of steps
			# clean and save rl_output to state
			rhos = self.clean_rl_output(rl_output)
			if state is None:
				state = {}
			state['rl_output'] = rhos.copy()
			state['rhos'] = rhos.copy()
			# take action 1 - set rho values
			self._actor.step(state)
			# take action 2 - navi
			actions = self._navi._model.predict(self._navi_obs)
			navi_obs, total_reward, done, truncated, navi_state = self._navi.step(actions, state)
			state['navi_state'] = navi_state
			self._navi_obs = navi_obs.copy()
			aux_obs = np.concatenate([navi_obs, (np.array(actions)+ 1)/2, rhos])
			if done:
				self.end(state)
		except msgpackrpc.error.TimeoutError as e:
			utils.speak('*** crashed **')
			self.handle_crash()
			utils.speak('*** recovered **')
			# this is a hot fix, until SB3 has a way to remove erroneous steps
			# however in the grand scheme this one step on the replay buffer shouldnt have such a drastic affect
			aux_obs = self._null_data # fill with erroneous data (Zeros)
			total_reward = 0 # 0 reward
			done = True # finish this episode
			truncated = True
		# state is passed to stable-baselines3 callbacks
		return aux_obs, total_reward, done, truncated, state

	# called at beginning of each episode to prepare for next
	# returns first observation for new episode
	# spawn_to will overwrite previous spawns and force spawn at that x,y,z,yaw
	def reset(self, state = None, seed=None):
		if state is None:
			state = {}
		self.episode_counter += 1
		try_again = True
		while(try_again):
			try:
				# reset navi env
				navi_obs, navi_state = self._navi.reset()
				state['navi_state'] = navi_state
				self._navi_obs = navi_obs.copy()
				actions = [0] * len(self._navi._actor._actions)
				rhos = [1] * len(self._actor._actions)
				aux_obs = np.concatenate([navi_obs, (np.array(actions)+ 1)/2, rhos])
				self._null_data = np.zeros(aux_obs.shape).astype(aux_obs.dtype)
				try_again = False
			except msgpackrpc.error.TimeoutError as e:
				utils.speak('*** crashed **')
				self.handle_crash()
				utils.speak('*** recovered **')
		return aux_obs, state


	# called at the end of each episode for any clean up, when done=True
	# normally only reset() is used in OpenAI Gym environments
	# but end() is much easier/clear/necessary for modifiers and multiple envs
	# off-by-one errors aggregate when switching between multiple envs
	def end(self, state=None):
		try_again = True
		while(try_again):
			try:
				# end all components
				self._model.end(state)
				self._actor.end(state)
				try_again = False
			except msgpackrpc.error.TimeoutError as e:
				utils.speak('*** crashed **')
				self.handle_crash()
				utils.speak('*** recovered **')