from environments.environment import Environment
from component import _init_wrapper
from gym import spaces
import numpy as np
import utils

# environment that is heart of drone_rl - for training rl algos
class DroneRL(Environment):
    # even though we do not render, this field is necesary for sb3
    metadata = {"render.modes": ["rgb_array"]}

    # checkpoint every number of episodes - make evaluations and save model
    @_init_wrapper
    def __init__(self, drone_component, actor_component, observer_component, rewarder_component, terminators_components, 
                        others_components=[], show_states=True):
        super().__init__()
        self._nSteps = 0
        self._nEpisodes = 0
        self._evaluating = False

    def connect(self):
        super().connect()
        # even though we do not directly use the observation or action space, these fields are necesary for sb3
        self.observation_space = spaces.Box(0, 255, shape=self._observer._output_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actor._actions))
            
    # activate needed components
    def step(self, rl_output):
        state = {'rl_output':float(rl_output)}
        # take action
        transcribed_action = self._actor.act(rl_output)
        state['transcribed_action'] = transcribed_action
        # get observation
        observation = self._observer.observe()
        state['observation_component'] = observation._name
        # set state kinematics variables
        state['drone_position'] = self._drone.get_position() 
        state['yaw'] = self._drone.get_yaw() 
        # set rewards in state dictionary
        self._rewarder.reward(state)
        # check for termination
        self._nSteps += 1
        done = False
        for terminator in self._terminators:
            done = done or terminator.terminate(state)
        state['done'] = done
        if self._evaluating:
            state['nEpisodes'] = 'evaluation'
            observation.write()
        else:
            state['nEpisodes'] = self._nEpisodes
        state['nSteps'] = self._nSteps 
        # display state?
        #if self.show_states:
        #    Environment.show_state(state)
        # any other misc components
        for other in self._others:
            other.step(state)
        # state is passed to stable-baselines3 callbacks
        return observation.to_numpy(), state['total_reward'], done, state

    # called at end of episode to prepare for next, when step() returns done=True
    # returns first observation for new episode
    def reset(self):
        print('RESET Episode', self._nEpisodes, 'Evaluating?', self._evaluating)
        if not self._evaluating:
            self._nEpisodes += 1
        self._nSteps = 0
        self._drone.reset()
        for other in self._others:
            other.reset()
        self._actor.reset()
        self._observer.reset()
        self._rewarder.reset()
        for terminator in self._terminators:
            terminator.reset()
        self._drone.take_off()
        return self._observer.observe().to_numpy()