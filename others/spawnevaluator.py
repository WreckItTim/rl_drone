from others.other import Other
from component import _init_wrapper
from models.model import Model
import numpy as np

# objective is set x-meters in front of drone and told to go forward to it
class SpawnEvaluator(Other):
    # distance is meters in front point is set, spawns is optionally a tupple of spawn [position (x,y,z), yaw (degrees clockwise)] pairs
    @_init_wrapper
    def __init__(self, model_component, drone_component, evaluate_every_nEpisodes=10, distance=100, nTimes=1, spawns=None):
        self._nEpisodes = 0
        self._evaluating = False
        if spawns is not None:
            self._nSpawns = 0

    def spawn(self):
        if self.spawns is not None:
            spawn_index = self._nSpawns % len(self.spawns)
            position = np.array(self.spawns[spawn_index][0], dtype=float)
            yaw = self.spawns[spawn_index][1]
            self._drone.teleport(position)
            self._drone.set_yaw(yaw)
            self._nSpawns += 1

    def step(self, state):
        if self._evaluating:
            self._nSteps += 1
            self._states[self._nSteps] = state

    def reset(self):
        if self._evaluating:
            pass
        else:
            self._nEpisodes += 1
            if self._nEpisodes % self.evaluate_every_nEpisodes == 0:
                print('EVALUATE')
                self._evaluating = True
                for n in range(self.nTimes):
                    self.spawn()
                    self._nSteps = 0
                    self._states = {}
                    self._model.evaluate(self._model._environment)
                self._evaluating = False
                print('LEARN')