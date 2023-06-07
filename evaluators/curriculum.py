from evaluators.evaluator import Evaluator
from component import _init_wrapper
import os
import pickle
import random
import rl_utils as utils

# designed to be parallel
# evaluator will handle evaluations and decide to advance level
# trainer will handle checking if advanced to next level
# can be both if not parallel

# evaluator messages:
# 0 = empty
# 1 = finished evaluation , next_level?

# trainer messages:
# 0 = empty
# 1 = ready for evalutation , level_steps (-1 for final form), model_path
class Curriculum(Evaluator):
	@_init_wrapper
	def __init__(self,
				models_directory,
				model_component,
				eval_freq=100,

				is_evaluator=True,
				evaluator_fvar_component=None,
				evaluate_environment_component=None,
				nEpisodes=100,
				criteria=99,
				eval_counter=0,

				is_trainer=True,
				trainer_fvar_component=None,
				train_spawn_component=None,
				steps=[-1],
				in_final_form=False,
				level=0,
				level_steps=-1,
				ignore_first=False, # ignore level up from 0th eval
			): 
		pass


	# shared methods
	def reset_learning(self, state=None):
		self.in_final_form = False
		self.level = 0
		self.level_steps = -1
		self.ignore_first = False
		self.eval_counter = 0

	def update(self):
		if self.is_trainer:
			self.check_trainer()
		if self.is_evaluator:
			self.check_evaluator()			

	# evaluator methods
	def check_evaluator(self):
		msg = self._trainer_fvar.listen()
		# ready for evaluation
		if msg[0] == 1:
			self.level_steps = msg[1]
			model_path = msg[2]
			self._model.load_models(self.models_directory + model_path)
			self.evaluate_set()

	def evaluate_set(self):
		#utils.speak('evaluate_set()')
		# loop through all episodes
		aSuccesses = []
		lSuccesses = []
		for i in range(self.nEpisodes):
			# step through next episode
			success, astar_steps = self.evaluate_episode()
			aSuccesses.append(success)
			if astar_steps == self.level_steps:
				lSuccesses.append(success)
		self.eval_counter += 1
		aSuccess = round(100.*sum(aSuccesses)/len(aSuccesses),2)
		msg = [1, True]
		lSuccess = 'N/A'
		if len(lSuccesses) > 0:
			lSuccess = round(100.*sum(lSuccesses)/len(lSuccesses),2)
			if lSuccess < self.criteria:
				msg = [1, False]
		utils.speak(f'evaluation:{self.eval_counter} total_success:{aSuccess}% level_success:{lSuccess}%')
		self._evaluator_fvar.speak(msg)

	# steps through one evaluation episode
	def evaluate_episode(self):
		# start environment, returning first observation
		observation_data, first_state = self._evaluate_environment.start()
		astar_steps = first_state['astar_steps']
		# start episode
		done = False
		while(not done):
			# get rl output
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
			if done:
				success = state['termination_result'] == 'success'
		return success, astar_steps



	# trainer methods
	def check_trainer(self):
		next_level = False
		msg = self._evaluator_fvar.listen()
		if msg[0] == 1:
			next_level = msg[1]

		if self.in_final_form:
			self._spawn.nSteps = random.choice(self.steps)
		else:
			if next_level:
				if self.ignore_first:
					self.ignore_first = False
				else:
					self.level_up()

		if self.is_evaluator:
			if self._model.nEpisodes % self.eval_freq == 0:
				self.prep_next()
		elif msg[0] == 1:
			self.prep_next()

	def prep_next(self):
		level_steps = -1 if self.in_final_form else self._train_spawn.nSteps
		model_path = 'train_eps_' + str(self._model.nEpisodes) + '/'
		self._model.save_models(self.models_directory + model_path)
		msg = [1, level_steps, model_path]
		self._trainer_fvar.speak(msg)

	# goes to next level of curric learning
	def level_up(self):
		self.level += 1
		utils.speak(f'LEVEL UP:{self.level}')
		if self.level >= len(self.steps):
			self.in_final_form = True
			self.use_final_form()
		else:
			self._train_spawn.nSteps = self.steps[self.level]