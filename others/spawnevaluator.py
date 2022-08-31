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
                    ,_write_folder=None, nEvaluations=0):
        if spawns is not None:
            self._nSpawns = 0
        if _write_folder is None:
            self._write_folder = utils.get_global_parameter('write_folder') + 'evaluations/'
        if not os.path.exists(self._write_folder):
            os.makedirs(self._write_folder)
        self._train_episode = 0
        self._evaluation_episode = -1

    def spawn(self):
        if self.spawns is not None:
            spawn_index = self._evaluation_episode % len(self.spawns)
            position = np.array(self.spawns[spawn_index][0], dtype=float)
            yaw = self.spawns[spawn_index][1]
            self._drone.teleport(position)
            self._drone.set_yaw(yaw)

    def step(self, state):
        # save states while evaluating
        if self._environment._evaluating:
            freeze_state = state.copy()
            self._states[self._evaluation_episode][self._nSteps] = freeze_state
            self._nSteps += 1
            print('episode', self._evaluation_episode, 'yaw', self._drone.get_yaw())
                
    def reset(self):
        # handle resets while training - check when to do next set of evaluations
        if not self._environment._evaluating:
            if self._train_episode % self.evaluate_every_nEpisodes == 0:
                self._model.evaluate(self._model._environment, n_eval_episodes=self.nTimes)
            self._train_episode += 1

        # handle resets while evaluating
        if self._environment._evaluating:
            self._evaluation_episode += 1
            
            # start of all evaluations stuff here:
            if self._evaluation_episode == 0:
                self._states = {}
            
            # begin of a new evaluation episode stuff here:
            if self._evaluation_episode >= 0 and self._evaluation_episode < self.nTimes:
                self.spawn()
                self._nSteps = 0
                self._states[self._evaluation_episode] = {}
            
            # end of all evaluations stuff here:
            if self._evaluation_episode == self.nTimes:
                utils.write_json(self._states, self._write_folder + str(self.nEvaluations) + '.json')
                self.nEvaluations += 1
                self._evaluation_episode = -1 # reset to -1 for future evaluation sets