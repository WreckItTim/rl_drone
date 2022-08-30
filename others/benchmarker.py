from others.other import Other
from component import _init_wrapper, benchmark_components, get_all_components
import utils

# objective is set x-meters in front of drone and told to go forward to it
class BenchMarker(Other):
    # distance is meters in front point is set, spawns is optionally a tupple of spawn [position (x,y,z), yaw (degrees clockwise)] pairs
    @_init_wrapper
    def __init__(self, environment_component, benchmark_every_nEpisodes=10, _write_path=None):
        self._nEpisodes = 0
        if _write_path is None:
            self._write_path = utils.get_global_parameter('write_folder') + 'benchmarks.json'

    def reset(self):
        if not self._environment._evaluating:
            self._nEpisodes += 1
            if self._nEpisodes % self.benchmark_every_nEpisodes == 0:
                print('BENCHMARK COMPONENTS')
                benchmark_components(get_all_components(), write_path=self._write_path)