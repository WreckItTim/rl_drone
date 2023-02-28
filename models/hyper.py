from models.model import Model
from stable_baselines3 import DQN as sb3DQN
from component import _init_wrapper
from hyperopt import fmin, tpe, hp
from stable_baselines3 import TD3 as sb3TD3
import rl_utils as utils
import pandas as pd
import numpy as np
import os

# hyper optimiztion, uses hyperopt lib for gaussian search
# this hyper search optimizes max goal ditance traveled
# adjust best_score for your needs
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
			  write_folder = None, # writes hyper results to
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
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory') + self._name + '/'
		if not os.path.exists(self.write_folder):
			os.makedirs(self.write_folder)

		
	# set model write prefixes
	# to be used by save modifiers
	def write_prefix(self):
		prefix = 'iter' + str(self._iter_count) + '_'
		prefix += 'run' + str(self._n) + '_'
		return prefix
		
	# hyperopt objective to minimize
	def objective(self, params):
		scores = []
		self._iter_count += 1
		for n in range(self.nRuns):
			self._n = n
			utils.speak(f'HYPER iter:{self._iter_count} run:{n}')
			
			# reset components and vars for new learning loop
			self._configuration.reset_all()

			# set model hyper parameters
			model_arguments = self.default_params.copy()
			if 'learning_rate' in params:
				model_arguments['learning_rate'] = 10**int(-1*params['learning_rate'])
			if 'learning_starts' in params:
				model_arguments['learning_starts'] = int(params['learning_starts'])
			if 'tau' in params:
				model_arguments['tau'] = float(params['tau'])
			if 'buffer_size' in params:
				model_arguments['buffer_size'] = int(params['buffer_size'])
			if 'gamma' in params:
				model_arguments['gamma'] = float('.' + ''.join(['9' for _ in range(int(params['gamma']))]))
			if 'batch_size' in params:
				model_arguments['batch_size'] = int(params['batch_size'])
			if 'train_freq' in params:
				model_arguments['train_freq'] = (int(params['train_freq']), 'episode')
			if 'policy_delay' in params:
				model_arguments['policy_delay'] = int(params['policy_delay'])
			if 'target_policy_noise' in params:
				model_arguments['target_policy_noise'] = float(params['target_policy_noise'])
			if 'target_noise_clip' in params:
				model_arguments['target_noise_clip'] = float(params['target_noise_clip'])
			if 'policy_layers' in params and 'policy_nodes' not in params:
				model_arguments['policy_kwargs'] = {'net_arch':[64 for _ in range(int(params['policy_layers']))]}
			if 'policy_nodes' in params and 'policy_layers' not in params:
				model_arguments['policy_kwargs'] = {'net_arch':[int(params['policy_nodes']), int(params['policy_nodes'])]}
			if 'policy_layers' in params and 'policy_nodes' in params:
				model_arguments['policy_kwargs'] = {'net_arch':[int(params['policy_nodes']) for _ in range(int(params['policy_layers']))]}

			for param in model_arguments:
				if param == 'tensorboard_log':
					continue
				if param not in self._results_table:
					self._results_table[param] = []
				self._results_table[param].append(model_arguments[param])
			model_arguments['env'] = self._environment

			# make sb3model
			self._sb3model = self.sb3Type(**model_arguments)

			# run sb3 model (parent class will make calls to this)
			self._sb3model.learn(
				total_timesteps = self._total_timesteps,
				callback = self._callback,
				log_interval = self._log_interval,
				tb_log_name = self._tb_log_name,
				reset_num_timesteps = self._reset_num_timesteps,
			)

			# update table
			best_score = self._best_score # update from evaluator
			self._results_table['run' + str(n)].append(best_score)
			scores.append(best_score)

		# print table
		results_path = self.write_folder + '/hyper_results.csv'
		pdf = pd.DataFrame(self._results_table)
		pdf.to_csv(results_path)

		# return -mean of all runs (hyperopt minimizes the objective)
		return_val = -1.0 * np.mean(scores)
		return return_val


	# hyperopt
	def learn(self, 
		total_timesteps = 100_000,
		callback = None,
		log_interval = -1,
		tb_log_name = None,
		reset_num_timesteps = True,
		):
		
		# set params
		self._total_timesteps = total_timesteps
		self._callback = callback
		self._log_interval = log_interval
		self._tb_log_name = tb_log_name
		self._reset_num_timesteps = reset_num_timesteps
	
		# run Hyperopt - minimizing the objective function, with the given grid space, using TPE method, and 16 max iterations
		best = fmin(self.objective, self._space, algo=tpe.suggest, max_evals=self.max_evals)