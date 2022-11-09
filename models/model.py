# abstract class used to handle RL model
from component import Component
import utils
from os.path import exists
from component import _init_wrapper

class Model(Component):
	# WARNING: child init must set sb3Type, and should have any child-model-specific parameters passed through model_arguments
	# NOTE: env=None as training and evaluation enivornments are handeled by controller
	def __init__(self, model_path=None, replay_buffer_path=None, _model_arguments=None):
		self._is_hyper = False
		self._model_arguments = _model_arguments
		self._sb3model = None
		self.connect_priority = -1 # environment needs to connect first if creating a new sb3model

	def connect(self):
		super().connect()
		if self._is_hyper:
			pass
		self._model_arguments['env'] = self._environment
		# create model object if needs be
		_model_path = self.model_path
		if _model_path is not None and exists(_model_path):
			self.load(_model_path)
			self._sb3model.set_env(self._model_arguments['env'])
			print('loaded model from file')
		else:
			self._sb3model = self.sb3Type(**self._model_arguments)
		_replay_buffer_path = self.replay_buffer_path
		if _replay_buffer_path is not None and exists(_replay_buffer_path):
			self._sb3model.load_replay_buffer(_replay_buffer_path)
			print('loaded replay buffer from file')
		# set up model path to write to
		self.model_path = utils.get_global_parameter('working_directory') + 'model.zip'
		# set up model path to write to
		self.best_model_path = utils.get_global_parameter('working_directory') + 'best_model.zip'
		# set up replay buffer path to write to
		self.replay_buffer_path = utils.get_global_parameter('working_directory') + 'replay_buffer.pkl'

	def learn(self, 
		total_timesteps=10_000,
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
		# call sb3 learn method
		self._sb3model.learn(
			total_timesteps,
			callback = callback,
			log_interval= log_interval,
			tb_log_name = tb_log_name,
			eval_env = eval_env,
			eval_freq = eval_freq,
			n_eval_episodes = n_eval_episodes,
			eval_log_path = eval_log_path,
			reset_num_timesteps = reset_num_timesteps,
		)
		utils.speak('DONE LEARN')

	def dump(self, write_folder):
		if 'model' in self._dumps:
			self.save(self.model_path)
		if 'replay_buffer' in self._dumps:
			self.save_replay_buffer(self.replay_buffer_path)

	def predict(self, rl_output):
		rl_output, next_state = self._sb3model.predict(rl_output, deterministic=True)
		return rl_output
	
	# save sb3 model to path (sb3 auto appends file type at end)
	def save(self, path):
		self._sb3model.save(path)

	def save_best(self):
		self._sb3model.save(self.best_model_path)
		
	# save sb3 replay buffer to path (sb3 auto appends file type at end)
	def save_replay_buffer(self, path):
		if self._has_replay_buffer:
			self._sb3model.save_replay_buffer(path)

	# load sb3 model from path, must set sb3Load from child
	def load(self, path):
		if not exists(path):
			utils.error(f'invalid Model.load() path:{path}')
		else:
			self._sb3model = self.sb3Load(path)

	# when using the debug controller
	def debug(self):
		self.learn()
		self.evaluate(self._environment)
