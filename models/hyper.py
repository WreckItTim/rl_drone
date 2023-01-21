# hyper optimiztion
from models.model import Model
from stable_baselines3 import DQN as sb3DQN
from component import _init_wrapper
from hyperopt import fmin, tpe, hp
from stable_baselines3 import TD3 as sb3TD3
import utils
import pandas as pd
import numpy as np

class Hyper(Model):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  environment_component,
			  _space,
			  model_type,
			  default_params={},
			  max_evals = 16,
			  nRuns = 1,
		):
		super().__init__()
		self._is_hyper = True
		self._space = _space
		self._iter_count = 0
		self._best_goal = 0
		if self.model_type == 'TD3':
			self.sb3Type = sb3TD3
			self.sb3Load = sb3TD3.load
			self._has_replay_buffer = True
		if self.model_type == 'DQN':
			self.sb3Type = sb3DQN
			self.sb3Load = sb3DQN.load
			self._has_replay_buffer = True
		self._results_table = {}
		for n in range(nRuns):
			self._results_table['run' + str(n)] = []

	def connect(self):
		super().connect()
	
	# hyperopt objective to minimize
	def objective(self, params):
		scores = []
		self._iter_count += 1
		for n in range(self.nRuns):
			utils.speak('HYPER iter:{self._iter_count} run:{n}')
			
			# reset components and vars for new learning loop
			self._configuration.reset_all()

			# set model hyper parameters
			model_arguments = self.default_params.copy()
			model_arguments.update(params)
			if 'learning_rate' in params:
				model_arguments['learning_rate'] = 10**int(-1*params['learning_rate'])
			if 'gamma' in params:
				model_arguments['gamma'] = float('.' + ''.join(['9' for _ in range(int(params['gamma']))]))
			for param in model_arguments:
				if param not in self._results_table:
					self._results_table[param] = []
				self._results_table[param].append(model_arguments[param])
			model_arguments['env'] = self._environment

			# make sb3model
			self._sb3model = self.sb3Type(**model_arguments)

			# set write prefixes
			self._write_folder = utils.get_global_parameter('working_directory') + 'Hyper/'
			self._write_folder += 'iter' + str(self._iter_count) + '_'
			self._write_folder += 'run' + str(self.n) + '_'

			# run sb3 model (parent class will make calls to this)
			self._sb3model.learn(
				total_timesteps = self._total_timesteps,
				callback = self._callback,
				log_interval = self._log_interval,
				tb_log_name = self._tb_log_name,
				reset_num_timesteps = self._reset_num_timesteps,
			)

			# update table
			self._results_table['run' + str(n)].append(self._score)
			scores.append(self._score)

		# print table
		results_path = utils.get_global_parameter('working_directory') +  + 'Hyper/hyper_results.csv'
		pdf = pd.DataFrame(self._results_table)
		pdf.to_csv(results_path)

		# return -mean of all runs (hyperopt minimizes the objective)
		return_val = -1.0 * np.mean(scores)
		return return_val


	# hyperopt
	def learn(self, 
		total_timesteps = 10_000,
		callback = None,
		log_interval = -1,
		tb_log_name = None,
		reset_num_timesteps = False,
		):
		
		# set params
		self._total_timesteps = total_timesteps
		self._callback = callback
		self._log_interval = log_interval
		self._tb_log_name = tb_log_name
		self._reset_num_timesteps = reset_num_timesteps
	
		# run Hyperopt - minimizing the objective function, with the given grid space, using TPE method, and 16 max iterations
		best = fmin(self.objective, self._space, algo=tpe.suggest, max_evals=self.max_evals)