from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import rl_utils as utils
import os

# an environment is the heart of RL algorithms
# the Goal flavor wants the drone to go to Point A to Point B
# Aux adds slimmable methods
# steps until termination then resets
# if a saver modifier is used, will write observations and states
class AuxEnv(Environment):

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
	def start(self, state = None):
		self.episode_counter += 1
		# reset navi env
		navi_obs = self._navi.start()
		self._navi_obs = navi_obs.copy()
		actions = [0] * len(self._navi._actor._actions)
		rhos = [1] * len(self._actor._actions)
		aux_obs = np.concatenate([navi_obs, (np.array(actions)+ 1)/2, rhos])
		return aux_obs