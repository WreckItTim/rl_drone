from others.other import Other
from component import _init_wrapper
import utils

# objective is set x-meters in front of drone and told to go forward to it
class Saver(Other):
	@_init_wrapper
	def __init__(self, 
			  model_component, 
			  environment_component,
			  nEpisodes=0, 
			  save_every_nEpisodes=10, 
			  save_model=True,
			  save_replay_buffer=True,
			  save_configuration_file=True,
			  save_benchmarks=True,
			  _write_folder=None
			  ):
		if _write_folder is None:
			self._write_folder = utils.get_global_parameter('write_folder')

	def save(self):
		print('SAVE')
		if self.save_model:
			self._model.save(self._write_folder + 'model')
		if self.save_replay_buffer:
			self._model.save_replay_buffer(self._write_folder + 'replay_buffer')
		if self.save_configuration_file:
			self._configuration.save(self._write_folder + 'configuration.json')
		if self.save_benchmarks:
			self._configuration.log_benchmarks(self._write_folder + 'benchmarks.json')

	def reset(self):
		if not self._environment._evaluating:
			self.nEpisodes += 1
			if self.nEpisodes % self.save_every_nEpisodes == 0:
				self.save()

	# when using the debug controller
	def debug(self):
		self.save()