from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import rl_utils as utils
import os
from stable_baselines3.common.type_aliases import TrainFreq
from gym import spaces

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
				is_evaluation_env=False,
				step_counter=0, 
				episode_counter=0, 
				start = 100, # turn off noise
		):
		super().__init__()

	# just makes the rl_output from SB3 more readible
	def clean_rl_output(self, rl_output):
		if np.issubdtype(rl_output.dtype, np.integer):
			return int(rl_output)
		if np.issubdtype(rl_output.dtype, np.floating):
			return rl_output.astype(float).tolist()

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
		navi_obs, navi_reward, navi_done, navi_state = self._navi.step(actions, state)
		self._navi_obs = navi_obs.copy()
		aux_obs = np.concatenate([navi_obs, (np.array(actions)+ 1)/2, rhos])
		# state is passed to stable-baselines3 callbacks
		return aux_obs, navi_reward, navi_done, navi_state

	# called at beginning of each episode to prepare for next
	# returns first observation for new episode
	# spawn_to will overwrite previous spawns and force spawn at that x,y,z,yaw
	def reset(self, state = None):
		self.episode_counter += 1
		if not self.is_evaluation_env and self.episode_counter > self.start:
			self._model._sb3model.action_noise = None 
		# reset navi env
		navi_obs = self._navi.reset()
		self._navi_obs = navi_obs.copy()
		actions = [0] * len(self._navi._actor._actions)
		rhos = [1] * len(self._actor._actions)
		aux_obs = np.concatenate([navi_obs, (np.array(actions)+ 1)/2, rhos])
		return aux_obs