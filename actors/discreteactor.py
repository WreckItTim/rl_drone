# discrete actions - RL returns an index specifying which action to take
from actors.actor import Actor
from random import randint
from component import _init_wrapper
import random
from gym import spaces

class DiscreteActor(Actor):
	# constructor
	@_init_wrapper
	def __init__(self, actions_components=[]):
		super().__init__()

	# interpret action from RL
	def act(self, rl_output):
		self._actions[rl_output].act()
		return self._actions[rl_output]._name

	# returns dioscrete action space
	def get_space(self):
		return spaces.Discrete(len(self._actions))
		
	# when using the debug controller
	def debug(self):
		rl_output = random.randint(0, len(self._actions)-1)
		self.act(rl_output)