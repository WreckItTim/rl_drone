# abstract class used to handle observations to input into rl algo
from component import Component
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from gym import spaces

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

	def connect(self):
		super().connect()
		# even though we do not directly use the observation or action space, these fields are necesary for sb3
		self.observation_space = self._observer.get_space()
		self.action_space = self._actor.get_space()

	# when using the debug controller
	def debug(self):
		# reset environment
		self.reset()
		# get first observation
		observation_numpy = self._observer.observe().to_numpy()
		# sample rl output
		rl_output = self.action_space.sample()
		# take a step
		observation_numpy, reward, done, state = self._evaluate_environment.step(rl_output)

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