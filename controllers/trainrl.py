# trains a reinforcment learning algorithm
from controllers.controller import Controller
from component import _init_wrapper

class TrainRL(Controller):
	# constructor
	@_init_wrapper
	def __init__(self, 
				 model_component,
				 environment_component,
				 evaluator_component = None,
				 total_timesteps = 1_000_000,
				 callback = None,
				 log_interval = -1,
				 tb_log_name = None,
				 eval_env = None,
				 eval_freq = -1,
				 n_eval_episodes = -1,
				 eval_log_path = None,
				 continue_training=True,
				 ):
		super().__init__()

	# runs control on components
	def run(self):
		if self.continue_training:
			# to continually train model from same learning loop
			_num_timesteps = self._environment.step_counter
			_episode_num = self._environment.episode_counter
			self._model._sb3model.num_timesteps = _num_timesteps
			self._model._sb3model._episode_num = _episode_num
			_total_timesteps = self.total_timesteps - _num_timesteps
			_reset_num_timesteps = False
		else:
			# train model from new learning loop (will use current weights new or old)
			_total_timesteps = self.total_timesteps
			_reset_num_timesteps = True
			self._environment.episode_counter = 0
			self._environment.step_counter = 0
			if self._evaluator is not None:
				self._evaluator.reset_stopping()
		# learn baby learn
		self._model.learn(
			total_timesteps = _total_timesteps,
			callback = self.callback,
			log_interval = self.log_interval,
			tb_log_name = self.tb_log_name,
			eval_env = self.eval_env,
			eval_freq = self.eval_freq,
			n_eval_episodes = self.n_eval_episodes,
			eval_log_path = self.eval_log_path,
			reset_num_timesteps = _reset_num_timesteps,
			)