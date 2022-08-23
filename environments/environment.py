# abstract class used to handle observations to input into rl algo
from component import Component, get_component
from matplotlib.pyplot import imshow, show

# OpenAI Gym enviornment needed to run Stable_Baselines3
class Environment(Component):
    @staticmethod
    def show_state(state):
        action = state['transcribed_action']
        x = int(state['drone_position'][0])
        y = int(state['drone_position'][1])
        z = int(state['drone_position'][2])
        distance = int(state['distance'])
        episode = state['nEpisodes']
        step = state['nSteps']
        total_reward = round(state['total_reward'], 4)
        rewards = []
        for key in state:
            if 'reward_from_' in key:
                rewards.append(round(state[key], 4))
        print(f'episode:{episode} step:{step} action:{action}  position:({x},{y},{z})  distance:{distance}  total_reward:{total_reward}  rewards:{rewards}') 

    # constructor
    def __init__(self):
        pass

    def activate(self):
        observation_array = self.reset()
        imshow(observation_array, cmap='gray')
        show()
        for step in range(2):
            observation_array, reward, done, state = self.step(self.action_space.sample())
            imshow(observation_array, cmap='gray')
            print('printing state ... ', state)
            show()
        observation_array = self.reset()
        imshow(observation_array, cmap='gray')
        show()

    ## methods that are expected to be defined and called from OpenAI Gym and Stable_Baselines3

    # step called after observation and input action to take
    # take action then create next state to progress for next step
    # must return observation, reward, done, info
        # observation - input to rl model after taken action 
        # reward - calcuated reward at state after taken action
        # done - True or False if the episode is done or not 
        # info - auxilary diction of info for whatever
        
    def step(self, rl_output):
        raise NotImplementedError

    # called at end of episode to prepare for next, when step() returns done=True
    # returns first observation for new episode
    def reset(self):
        raise NotImplementedError