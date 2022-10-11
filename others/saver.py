from others.other import Other
from component import _init_wrapper
import utils

# objective is set x-meters in front of drone and told to go forward to it
class Saver(Other):
	@_init_wrapper
	def __init__(self, 
			  model_component=None, 
			  environment_component=None,
			  frequency=10, 
			  save_model=True,
			  save_replay_buffer=True,
			  save_configuration_file=True,
			  save_benchmarks=True,
			  write_folder=None
			  ):
		if write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')

	def save(self):
		print('SAVE')
		if self.save_model:
			self._model.save(self.write_folder + 'model')
		if self.save_replay_buffer:
			self._model.save_replay_buffer(self.write_folder + 'replay_buffer')
		if self.save_configuration_file:
			self._configuration.save(self.write_folder + 'configuration.json')
		if self.save_benchmarks:
			self._configuration.log_benchmarks(self.write_folder + 'benchmarks.json')

	def reset(self):
		if self._environment.episode_counter % self.frequency == 0:
			self.save()

	# when using the debug controller
	def debug(self):
		self.save()