from others.other import Other
from component import _init_wrapper
import utils

# objective is set x-meters in front of drone and told to go forward to it
class ModelSaver(Other):
    # distance is meters in front point is set, spawns is optionally a tupple of spawn [position (x,y,z), yaw (degrees clockwise)] pairs
    @_init_wrapper
    def __init__(self, model_component, save_every_nEpisodes=10, write_path=None):
        self._nEpisodes = 0
        if write_path is None:
            self.write_path = utils.global_parameters['write_folder'] + 'model'

    def reset(self):
        self._nEpisodes += 1
        if self._nEpisodes % self.save_every_nEpisodes == 0:
            print('SAVE MODEL')
            self._model.save(self.write_path)