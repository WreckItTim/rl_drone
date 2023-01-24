# abstract class used to handle RL model
from component import Component
import utils
from os.path import exists

class Model(Component):
	# WARNING: child init must set sb3Type, and should have any child-model-specific parameters passed through model_arguments
		# child init also needs to save the training environment (make environment_component a constructor parameter)
	# NOTE: env=None as training and evaluation enivornments are handeled by controller
	def __init__(self, 
			  read_model_path=None, 
			  read_replay_buffer_path=None, 
			  _model_arguments=None
			  ):
		# if the model is a hyper parameter tuner, some things get handeled differently
		self._is_hyper = False
		self._model_arguments = _model_arguments
		self._sb3model = None
		self.connect_priority = -1 # environment needs to connect first if creating a new sb3model

	def connect(self):
		super().connect()
		if self._is_hyper:
			return
		self._model_arguments['env'] = self._environment
		# create model object
		if self.read_model_path is not None and exists(self.read_model_path):
			self.load_model(self.read_model_path)
			self._sb3model.set_env(self._model_arguments['env'])
			print('loaded model from file')
		else:
			self._sb3model = self.sb3Type(**self._model_arguments)
		# replay buffer init
		if self.read_replay_buffer_path is not None and exists(self.read_replay_buffer_path):
			self.load_replay_buffer(self.read_replay_buffer_path)
			print('loaded replay buffer from file')
			
	# this will toggle if to checkpoint model and replay buffer
	def set_save(self,
			  track_save,
			  track_vars=[
				  'model', 
				  'replay_buffer',
				  ],
			  ):
		self._track_save = track_save
		self._track_vars = track_vars.copy()
	
	# save sb3 model and replay_buffer to path
	# pass in write_folder to state
	def save(self, state):
		write_folder= state['write_folder']
		if 'model' in self._track_vars:
			self.save_model(write_folder + 'model.zip')
		if '_has_replay_buffer' in self._track_vars and self._has_replay_buffer:
			self.save_replay_buffer(write_folder +  + 'replay_buffer.zip')
	def save_model(self, path):
		self._sb3model.save(path)
	def save_replay_buffer(self, path):
		self._sb3model.save_replay_buffer(path)

	# load sb3 model from path, must set sb3Load from child
	def load_model(self, path):
		if not exists(path):
			utils.error(f'invalid Model.load_model() path:{path}')
		else:
			self._sb3model = self.sb3Load(path)
		
	# load sb3 replay buffer from path
	def load_replay_buffer(self, path):
		if not exists(path):
			utils.error(f'invalid Model.load_replay_buffer() path:{path}')
		elif self._has_replay_buffer:
			self._sb3model.load_replay_buffer(path)
		else:
			utils.warning(f'trying to load a replay buffer to a model that does not use one')

	# runs learning loop on model
	def learn(self, 
		total_timesteps=10_000,
		callback = None,
		log_interval = -1,
		tb_log_name = None,
		reset_num_timesteps = False,
		):
		# call sb3 learn method
		self._sb3model.learn(
			total_timesteps,
			callback = callback,
			log_interval= log_interval,
			tb_log_name = tb_log_name,
			reset_num_timesteps = reset_num_timesteps,
		)
		
	# makes a single prediction g;iven input data
	def predict(self, rl_input):
		rl_output, next_state = self._sb3model.predict(rl_input, deterministic=True)
		return rl_output
