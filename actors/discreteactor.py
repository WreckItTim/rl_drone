from actors.actor import Actor
from random import randint
from component import _init_wrapper
import random
import utils
from gym import spaces

# handles discrete actions - RL returns an index specifying which action to take
class DiscreteActor(Actor):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  actions_components=[],
			  ):
		super().__init__()

	# interpret action from RL
	def step(self, state):
		rl_output = state['rl_output']
		self._actions[rl_output].step(state)
		state['transcribed_action'] = self._actions[rl_output]._name
		
	# take random action from list unless specified
	def debug(self, state=None):
		if state is None:
			sampled_output = random.randint(0, len(self._actions)-1)
			state = {
				'rl_output': sampled_output,
			}
		else:
			sampled_output = state['rl_output']
		utils.speak(f'taking discrete action at index {sampled_output}...')
		self.step(state)


	# returns dioscrete action space of type Discrete
	def get_space(self):
		return spaces.Discrete(len(self._actions))