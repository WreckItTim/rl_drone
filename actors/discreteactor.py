from actors.actor import Actor
from random import randint
from component import _init_wrapper
import random
import rl_utils as utils
from gym import spaces

# handles discrete actions - RL returns an index specifying which action to take
class DiscreteActor(Actor):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  actions_components=[],
			  ):
		super().__init__()
		self._type = 'discrete'

	# interpret action from RL
	def step(self, state):
		rl_output = state['rl_output']
		self._actions[rl_output].step(state)
		state['transcribed_action'] = self._actions[rl_output]._name
		
	# take random action from list unless specified
	def debug(self, state=None):
		while True:
			index = utils.prompt(f'Enter \'b\' to go back to parent debug menu. Enter a valid index of action to activate. Enter \'r\' to randomly sample action...')
			if index == 'b':
				break
			if index == 'r':
				index = random.randint(0, len(self._actions)-1)
			else:
				try:
					index = int(index)
					if index < 0 or index >= len(self._actions):
						print('invalid entry')
						continue
				except ValueError:
					print('invalid entry')
					continue
			state = {
				'rl_output': index,
			}
			_drone = self._configuration.get_component('Drone')
			position_before = _drone.get_position()
			yaw_before = _drone.get_yaw()
			collision_before = _drone.check_collision()
			utils.speak(f'before action... position={position_before} yaw={yaw_before} collision={collision_before}')
			action_name = self._actions[index]._name
			utils.speak(f'taking discrete action {action_name} at index {index}...')
			self.step(state)
			position_after = _drone.get_position()
			yaw_after = _drone.get_yaw()
			collision_after = _drone.check_collision()
			utils.speak(f'after action... position={position_after} yaw={yaw_after} collision={collision_after}')


	# returns dioscrete action space of type Discrete
	def get_space(self):
		return spaces.Discrete(len(self._actions))