from environemnt import Enivironment
from component import _init_wrapper

# OpenAI Gym enviornment needed to run Stable_Baselines3
class DronetRL(Enivironment):

    @_init_wrapper
    def __init__(self, component_names=[], name=None):
        super().__init__()

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
        for component in self._components:
            component.step(state)
        observation, reward, done = state['observation'], state['reward'], state['done']
        return observation.to_numpy(), reward, done, {}

    # called at end of episode to prepare for next, when step() returns done=True
    # returns first observation for new episode
    def reset(self):
        state = {}
        for component in self._components:
            component.reset(state)
        return state['observation']