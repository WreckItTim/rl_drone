from others.other import Other
from component import _init_wrapper
import utils

# objective is set x-meters in front of drone and told to go forward to it
class ReplayBufferSaver(Other):
	# distance is meters in front point is set, spawns is optionally a tupple of spawn [position (x,y,z), yaw (degrees clockwise)] pairs
	@_init_wrapper
	def __init__(self, model_component, environment_component, nEpisodes=0, save_every_nEpisodes=100, _write_path=None):
		if _write_path is None:
			self._write_path = utils.get_global_parameter('write_folder') + 'replay_buffer'

	def reset(self):
		if not self._environment._evaluating:
			self.nEpisodes += 1
			if self.nEpisodes % self.save_every_nEpisodes == 0:
				print('SAVE REPLAY BUFFER')
				self._model.save_replay_buffer(self._write_path)

	# when using the debug controller
	def debug(self):
		self._model.save_replay_buffer(self._write_path)