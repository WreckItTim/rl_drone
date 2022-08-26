from others.other import Other
from component import _init_wrapper
from models.model import Model
import numpy as np
import utils
import os

# objective is set x-meters in front of drone and told to go forward to it
class SpawnEvaluator(Other):
    # distance is meters in front point is set, spawns is optionally a tupple of spawn [position (x,y,z), yaw (degrees clockwise)] pairs
    @_init_wrapper
    def __init__(self, model_component, drone_component, environment_component, evaluate_every_nEpisodes=10, distance=100, nTimes=1, spawns=None
                    ,_write_folder=None):
        self._nEpisodes = -1
        self._evaluating = False
        if spawns is not None:
            self._nSpawns = 0
        if _write_folder is None:
            self._write_folder = utils.get_global_parameter('write_folder') + 'evaluations/'
        if not os.path.exists(self._write_folder):
            os.makedirs(self._write_folder)
        self._nEvaluation_sets = 0
        self._nEvaluations = -1

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
            freeze_state = state.copy()
            self._states[self._nEvaluations][self._nSteps] = freeze_state

    def reset(self):
        if self._evaluating and self._nEvaluations < self.nTimes:
            self._nEvaluations += 1
            self.spawn()
            self._nSteps = 0
            self._states[self._nEvaluations] = {}
        elif self._evaluating:
            self._environment._evaluating = False
        else:
            if self._nEpisodes == -1 or (self._nEpisodes % self.evaluate_every_nEpisodes == 0 and self._nEpisodes > 0):
                print('EVALUATE')
                self._evaluating = True
                self._environment._write_observations = True
                self._environment._evaluating = True
                self._states = {}
                self._nEvaluations = 0
                self._model.evaluate(self._model._environment, n_eval_episodes=self.nTimes)
                utils.write_json(self._states, self._write_folder + str(self._nEvaluation_sets) + '.json')
                self._nEvaluation_sets += 1
                self._environment._evaluating = False
                self._environment._write_observations = False
                self._evaluating = False
                print('LEARN')
            self._nEpisodes += 1