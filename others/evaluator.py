from others.other import Other
from component import _init_wrapper
from models.model import Model
import numpy as np
import utils
import os

# objective is set x-meters in front of drone and told to go forward to it
class Evaluator(Other):
	@_init_wrapper
	def __init__(self, 
			  model_component, 
			  drone_component, 
			  environment_component, 
			  spawners_components,
			  evaluate_every_nEpisodes=1000, 
			  _write_folder=None, 
			  nEvaluations=0
			  ):
		if _write_folder is None:
			self._write_folder = utils.get_global_parameter('write_folder') + 'evaluations/'
		if not os.path.exists(self._write_folder):
			os.makedirs(self._write_folder)
		self._train_episode = 0
		self._evaluation_episode = -1
		self._nSpawns = len(spawners_components)

	def step(self, state):
		# save states while evaluating
		if self._environment._evaluating:
			freeze_state = state.copy()
			self._states[self._evaluation_episode][self._nSteps] = freeze_state
			self._nSteps += 1

	def spawn(self):
		self._spawners[self._evaluation_episode].spawn()
				
	def reset(self):
		# handle resets while training - check when to do next set of evaluations
		if not self._environment._evaluating:
			if self._train_episode % self.evaluate_every_nEpisodes == 0:
				self._model.evaluate(self._model._environment, n_eval_episodes=self._nSpawns)
			self._train_episode += 1

		# handle resets while evaluating
		if self._environment._evaluating:
			self._evaluation_episode += 1
			
			# start of all evaluations stuff here:
			if self._evaluation_episode == 0:
				self._states = {}
			
			# begin of a new evaluation episode stuff here:
			if self._evaluation_episode >= 0 and self._evaluation_episode < self._nSpawns:
				self.spawn()
				self._nSteps = 0
				self._states[self._evaluation_episode] = {}
			
			# end of all evaluations stuff here:
			if self._evaluation_episode == self._nSpawns:
				utils.write_json(self._states, self._write_folder + str(self.nEvaluations) + '.json')
				self.nEvaluations += 1
				self._evaluation_episode = -1 # reset to -1 for future evaluation sets

	# when using the debug controller
	def debug(self):
		self.reset()