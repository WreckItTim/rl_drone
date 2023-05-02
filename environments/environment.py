# abstract class used to handle observations to input into rl algo
from component import Component

# follows OpenAI structure with some changes
class Environment(Component):

	# constructor
	def __init__(self):
		pass

	def connect(self, state=None):
		super().connect()

	# step called after observation and input action to take
	# take action then create next state to progress for next step
	# must return observation, reward, done, info
		# observation - input to rl model after taken action 
		# reward - calcuated reward at state after taken action
		# done - True or False if the episode is done or not 
	def step(self, rl_output):
		raise NotImplementedError

	# called at begin of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def start(self):
		pass

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def end(self):
		pass