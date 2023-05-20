from evaluators.evaluator import Evaluator
from component import _init_wrapper
import os
import pickle
import rl_utils as utils

class Static(Evaluator):
	@_init_wrapper
	def __init__(self,
				evaluate_environment_component, # environment to run eval in
				model_component,
				spawn_component = None,
				goal_component = None,
				distances = [], # distances to goal (calcs a spawn and goal for each ele)
				dz = 4, # fixed altitude 
				read_spawns_path = None,
				read_goals_path = None,
				write_spawns_path = None,
				write_goals_path = None,
				write_results_path = None,
			): 
		if write_results_path is not None:
			self.write_results_path = utils.fix_directory(write_results_path)
		if write_spawns_path is not None:
			self.write_spawns_path = utils.fix_directory(write_spawns_path)
			if not os.path.exists(self.write_spawns_path):
				os.makedirs(self.write_spawns_path)
		if write_goals_path is not None:
			self.write_goals_path = utils.fix_directory(write_goals_path)
			if not os.path.exists(self.write_goals_path):
				os.makedirs(self.write_goals_path)

	def connect(self, state=None):
		super().connect(state)
		read_spawns = self.read_spawns_path is not None
		read_goals = self.read_goals_path is not None
		if read_spawns:
			self._spawns = pickle.load(open(self.read_spawns_path, 'rb'))
		else:
			self._spawns = []
		if read_goals:
			self._goals = pickle.load(open(self.read_goals_path, 'rb'))
		else:
			self._goals = []
		for i, distance in enumerate(self.distances):
			if read_spawns:
				spawn = self._spawns[i]
			else:
				pos, yaw = self._spawn.get_spawn()
				spawn = pos + [yaw]
				self._spawns.append(spawn)
			if read_goals:
				goal = self._goals[i]
			else:
				goal = self._goal.get_goal(spawn[:3], spawn[3], r=distance, dz=self.dz)
				self._goals.append(goal)
		self._nEpisodes = len(self._spawns)
		if self.write_spawns_path is not None:
			pickle.dump(self._spawns, open(self.write_spawns_path, 'wb'))
		if self.write_goals_path is not None:
			pickle.dump(self._goals, open(self.write_goals_path, 'wb'))

	# steps through one evaluation episode
	def evaluate_episode(self, spawn, goal):
		start_state = {
			'spawn_to':spawn,
			'goal_at':goal,
		}
		# start environment, returning first observation
		observation_data = self._evaluate_environment.start(start_state)
		# start episode
		done = False
		while(not done):
			# get rl output
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, state = self._evaluate_environment.step(rl_output)
			if done:
				success = state['termination_result'] == 'success'
		# call end for modifiers
		self._model.end()
		return success

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# loop through all episodes
		successes = []
		for i in range(self._nEpisodes):
			# step through next episode
			successes.append(self.evaluate_episode(self._spawns[i], self._goals[i]))
		utils.speak('evaluated with %Success=', round(100.*sum(successes)/len(successes),2))
		if self.write_results_path is not None:
			fname = self.write_results_path + 'eps_' + str(self._model.nEpisodes) + '.p'
			pickle.dump(successes, open(fname, 'wb'))