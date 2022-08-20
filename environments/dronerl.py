from environments.environment import Environment
from component import _init_wrapper
from gym import spaces
import numpy as np


def show_state(state):
    action = state['transcribed_action']
    x = int(state['drone_position'][0])
    y = int(state['drone_position'][1])
    z = int(state['drone_position'][2])
    distance = int(state['distance'])
    total_reward = round(state['total_reward'], 4)
    rewards = []
    for key in state:
        if 'reward_from_' in key:
            rewards.append(round(state[key], 4))
    print(f'action:{action}  position:({x},{y},{z})  distance:{distance}  total_reward:{total_reward}  rewards:{rewards}')

# OpenAI Gym enviornment needed to run Stable_Baselines3
class DroneRL(Environment):
    metadata = {"render.modes": ["rgb_array"]}

    @_init_wrapper
    def __init__(self, drone_component, actor_component, observer_component, rewarder_component
                    , terminator_components=[], other_components=[], show_states=True):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, shape=self._observer.output_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actor._actions))

    ## methods that are expected to be defined and called from OpenAI Gym and Stable_Baselines3

    # step called after observation and input action to take
    # take action then create next state to progress for next step
    # must return observation, reward, done, info
        # observation - input to rl model after taken action 
        # reward - calcuated reward at state after taken action
        # done - True or False if the episode is done or not 
        # info - auxilary diction of info for whatever
    # everything is done from component
    def step(self, rl_output):
        state = {'rl_output':rl_output}
        transcribed_action = self._actor.act(rl_output)
        state['transcribed_action'] = transcribed_action
        observation = self._observer.observe()
        state['observation_component'] = observation._name
        self._rewarder.evaluate(state)
        done = False
        for terminator in self._terminators:
            done = done or terminator.evaluate(state)
        state['done'] = done
        for other in self._others:
            other.activate()
        if self.show_states:
            show_state(state)
        return observation.to_numpy(), state['total_reward'], done, state

    # called at end of episode to prepare for next, when step() returns done=True
    # returns first observation for new episode
    def reset(self):
        self._drone.reset()
        self._drone.take_off()
        self._actor.reset()
        self._observer.reset()
        self._rewarder.reset()
        for terminator in self._terminators:
            terminator.reset()
        for other in self._others:
            other.reset()
        return self._observer.observe().to_numpy()