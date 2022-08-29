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
        self._nEpisodes = 0
        if spawns is not None:
            self._nSpawns = 0
        if _write_folder is None:
            self._write_folder = utils.get_global_parameter('write_folder') + 'evaluations/'
        if not os.path.exists(self._write_folder):
            os.makedirs(self._write_folder)
        self._nEvaluations = 0
        self._set_finished = False
        self._states = {}

    def spawn(self):
        if self.spawns is not None:
            spawn_index = self._nSpawns % len(self.spawns)
            position = np.array(self.spawns[spawn_index][0], dtype=float)
            yaw = self.spawns[spawn_index][1]
            self._drone.teleport(position)
            self._drone.set_yaw(yaw)
            self._nSpawns += 1

    def step(self, state):
        # save states while evaluating
        if self._environment._evaluating:
            self._nSteps += 1
            freeze_state = state.copy()
            self._states[self._nEvaluations % self.nTimes][self._nSteps] = freeze_state
            # check if last step in episode
            if state['done']:
                self._nEvaluations += 1
                # if last reset, then log evaluations and prepare for next evaluation set (this is done in step() in case last reset is not called)
                if self._nEvaluations % self.nTimes == 0:
                    utils.write_json(self._states, self._write_folder + str(int(self._nEvaluations / self.nTimes)) + '.json')
                    self._states = {}
                    self._set_finished = True
                
    def reset(self):
        # handle resets while evaluating
        if self._environment._evaluating:
            self.spawn()
            self._nSteps = 0
            self._states[self._nEvaluations % self.nTimes] = {}
        # otherwise check when to evaluate
        if not self._environment._evaluating:
            if self._nEpisodes % self.evaluate_every_nEpisodes == 0:
                self._model.evaluate(self._model._environment, n_eval_episodes=self.nTimes)
            self._nEpisodes += 1