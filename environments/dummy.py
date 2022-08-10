
from gym import Env
import random

# OpenAI Gym enviornment needed to run Stable_Baselines3
class Dummy(Env):

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    ## methods that are expected to be defined and called from OpenAI Gym and Stable_Baselines3

    # step called after observation and input action to take
    # take action then create next state to progress for next step
    # must return observation, reward, done, info
        # observation - input to rl model after taken action 
        # reward - calcuated reward at state after taken action
        # done - True or False if the episode is done or not 
        # info - auxilary diction of info for whatever
    def step(self, action):
        obs = self.observation_space.sample()
        reward = random.random()
        done = random.choice([True, False, False, False])
        return obs, reward, done, {}

    # called at end of episode to prepare for next, when step() returns done=True
    # returns first observation for new episode
    def reset(self):
        return self.observation_space.sample()