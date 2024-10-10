from modifiers.modifier import Modifier
from component import _init_wrapper
import rl_utils as utils
import numpy as np

# checks against percent of training episodes resulting in success
class Curriculum(Modifier):
	@_init_wrapper
	def __init__(self,
				base_component, # componet with method to modify
				parent_method, # name of parent method to modify
				order, # modify 'pre' or 'post'?
				goal_component, # which component to level up
				level_up_amount, # will increase min,max goal distance by these many meters
				level_up_criteria, # percent of succesfull paths needed to level up
				level_up_buffer, # number of previous episodes to look at to measure level_up_criteria
				max_level,
				start_range = None,
				level = 1,
				track_record=[],
				track_idx=0,
				frequency = 1, # use modifiation after how many calls to parent method?
				counter = 0, # keepts track of number of calls to parent method
			): 
		super().__init__(base_component, parent_method, order, frequency, counter)
		self.connect_priority = -1 # goal component needs to connect first to get starting goal range
	
	def connect(self):
		super().connect()
		if self.start_range is None:
			self.start_range = self._goal.goal_range.copy()

	# reset learning loop 
	def reset_learning(self):
		self.reset_level()
		self.reset_track()
		self._goal.goal_range = self.start_range.copy()

	# modifier activate()	
	def activate(self, state):
		self.update_track(state)
		if self.evaluate(state):
			self._configuration.controller.stop()

	# evaluates all episodes for this next set
	def evaluate(self, state):
		if self.check_levelup(state):
			self.level_up()
		return self.check_terminate()


	## helper functions
	def update_track(self, state):
		self.track_record[self.track_idx] = 0 # overwrite to 0 to assum failure
		if state['reached_goal'] == True: 
			self.track_record[self.track_idx] = 1 # overwrite to 1 after observing success
		self.increment_track()
	def increment_track(self):
		self.track_idx = self.track_idx+1 if self.track_idx+1 < len(self.track_record) else 0   
	def reset_track(self):
		self.track_record = [0]*self.level_up_buffer
		self.track_idx = 0
	def reset_level(self):
		self.level = 1
	def level_up(self):
		last_range = np.array(self._goal.goal_range)
		next_range = (last_range + np.array(self.level_up_amount)).tolist()
		self._goal.goal_range = next_range
		self.level += 1
		self.reset_track()
		utils.speak(f'LEVELED UP!!! from {self.level-1} to {self.level}')
	def check_terminate(self):
		return self.level > self.max_level
	def check_levelup(self, state):
		total_success = sum(self.track_record)
		percent_success = total_success / len(self.track_record)
		termination_reason = state['termination_reason']
		reached_goal = state['reached_goal']
		utils.speak(f'termination:{termination_reason} goal:{reached_goal} level:{self.level} track_success:{percent_success}')
		return percent_success >= self.level_up_criteria