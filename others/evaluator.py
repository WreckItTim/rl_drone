from others.other import Other
from component import _init_wrapper
import rl_utils as utils
from configuration import Configuration

# checks against percent of training episodes resulting in success
class Evaluator(Other):
	@_init_wrapper
	def __init__(self, 
				goal_component, # used to get current goal
				curriculum_percent, # percent successfull episodes it takes to level up curriculum learning
				distance_max, # will train until succesfully reaches goal this far away
				level_up_distance, # amount to increase goal distance by, with curriculum learning on level up
				n_episodes = 100, # keep track of n-recent episode success
				track_record = [], # keep track of n-recent episode success
				track_idx = 0, # keep track of n-recent episode success
				distance_best = 0, # distance metric to maximize
				counter_eval = 1, # count number of evals
				verbose = 1, # handles logging output (0 silent, 1 eval progress)
				): 
		pass

	# returns progress of learning loop
	def get_progress(self):
		return self.distance_best / self.distance_max

	# reset learning loop to static values from connect()
	def reset_learning(self):
		self.distance_best = self._goal.static_r
		self.counter_eval = 0
		self.track_record = [0]*self.n_episodes
		self.track_idx = 0

	# reset learning loop to static values from connect()
	def level_up(self):
		self._goal.static_r += self.level_up_distance
		self._goal.random_r[0] += self.level_up_distance
		self._goal.random_r[1] += self.level_up_distance
		self.track_record = [0]*self.n_episodes
		self.track_idx = 0
        
	# evaluates all episodes for this next set
	def evaluate(self):
		# keep track of episode results
		total_success = sum(self.track_record)
		percent_success = total_success / len(self.track_record)
		level_up = percent_success >= self.curriculum_percent

		if self.verbose > 0:
			utils.speak(f'eval:{self.counter_eval} distance:{self._goal.static_r} success:{percent_success} level_up:{level_up}')

		stop = False # used to determine if we stop OUTTER TRAINING LOOP - typically shuts down entire program
		if level_up:
			self.distance_best = self._goal.static_r
			if self.distance_best >= self.distance_max:
				if self.verbose > 0:
					utils.speak(f'reached max distance, stopping training loop...')
				stop = True
			else:
				# amp up goal distance
				self.level_up()

		return stop
        
	# called at end of environment episode
	def end(self, state=None):
		self.track_record[self.track_idx] = 0    
		if state['termination_result'] == 'success': 
			self.track_record[self.track_idx] = 1 
		self.track_idx = self.track_idx+1 if self.track_idx+1 < len(self.track_record) else 0      
		
		stop = self.evaluate()
		self.counter_eval += 1
		if stop:
			self._configuration.controller.stop()  
