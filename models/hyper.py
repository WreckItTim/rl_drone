# hyper optimiztion
from models.model import Model
from stable_baselines3 import DQN as sb3DQN
from component import _init_wrapper
from hyperopt import fmin, tpe, hp
from stable_baselines3 import TD3 as sb3TD3
import utils
import pandas as pd

class Hyper(Model):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  environment_component,
			  _space,
			  model_type,
			  resets_components = None,
			  default_params={},
			  max_evals = 16,
		):
		super().__init__()
		self._is_hyper = True
		self._space = _space
		self._iter_count = 0
		self._best_goal = 0
		self._results_table = {
			'best_goal':[],
			}
		if self.model_type == 'TD3':
			self.sb3Type = sb3TD3
			self.sb3Load = sb3TD3.load
			self._has_replay_buffer = True
		if self.model_type == 'DQN':
			self.sb3Type = sb3DQN
			self.sb3Load = sb3DQN.load
			self._has_replay_buffer = True

	def connect(self):
		super().connect()
	
	# hyperopt objective to minimize
	def objective(self, params):
		# reset components and vars for new learning loop
		for component in self._resets:
			component.reset_learning()
		self._best_goal = 0
		self._iter_count += 1
		print('HYPER iter' + str(self._iter_count))
		model_arguments = self.default_params.copy()
		model_arguments.update(params)
		model_arguments['learning_rate'] = 10**int(-1*params['learning_rate'])
		model_arguments['gamma'] = float('.' + ''.join(['9' for _ in range(int(params['gamma']))]))
		for param in model_arguments:
			if param not in self._results_table:
				self._results_table[param] = []
			self._results_table[param].append(model_arguments[param])
		utils.write_json(model_arguments, utils.get_global_parameter('working_directory') + 'model_arguments_' + str(self._iter_count) + '.json')
		model_arguments['env'] = self._environment

		# make sb3model
		self._sb3model = self.sb3Type(**model_arguments)
		self.model_path = utils.get_global_parameter('working_directory') + 'model_' + str(self._iter_count) + '.zip'
		self.best_model_path = utils.get_global_parameter('working_directory') + 'best_model_' + str(self._iter_count) + '.zip'
		self.replay_buffer_path = utils.get_global_parameter('working_directory') + 'replay_buffer_' + str(self._iter_count) + '.pkl'

		# run sb3 model (parent class will make calls to this)
		self._sb3model.learn(
			total_timesteps = self._total_timesteps,
			callback = self._callback,
			log_interval = self._log_interval,
			tb_log_name = self._tb_log_name,
			eval_env = self._eval_env,
			eval_freq = self._eval_freq,
			n_eval_episodes = self._n_eval_episodes,
			eval_log_path = self._eval_log_path,
			reset_num_timesteps = self._reset_num_timesteps,
		)

		# update table
		self._results_table['best_goal'].append(self._best_goal)
		scores_path = utils.get_global_parameter('working_directory') + 'scores.csv'
		pdf = pd.DataFrame(self._results_table)
		pdf.to_csv(scores_path)

		return_val = -1.0 * self._best_goal
		return return_val


	# hyperopt
	def learn(self, 
		total_timesteps = 10_000,
		callback = None,
		log_interval = -1,
		tb_log_name = None,
		eval_env = None,
		eval_freq = -1,
		n_eval_episodes = -1,
		eval_log_path = None,
		reset_num_timesteps = False,
		):
		utils.speak('LEARN')
		
		# set params
		self._total_timesteps = total_timesteps
		self._callback = callback
		self._log_interval = log_interval
		self._tb_log_name = tb_log_name
		self._eval_env = eval_env
		self._eval_freq = eval_freq
		self._n_eval_episodes = n_eval_episodes
		self._eval_log_path = eval_log_path
		self._reset_num_timesteps = reset_num_timesteps
	
		# run Hyperopt - minimizing the objective function, with the given grid space, using TPE method, and 16 max iterations
		best = fmin(self.objective, self._space, algo=tpe.suggest, max_evals=self.max_evals)
		utils.speak('DONE LEARN')