# rewards based on number of steps taken in episode
from rewards.reward import Reward
from component import _init_wrapper
import math
import numpy as np

class Steps(Reward):
	# constructor
	@_init_wrapper
	def __init__(self,
			  max_steps=4,
			  update_steps=True, # if True, will add more steps for further distance to goal
			  step_ratio=1, # steps added per meter of distance to goal (added to initial max_steps)
			  terminate=True, # =True will terminate episodes when steps reached
			  ):
		super().__init__()
		self._max_steps = max_steps # make private to possibly update (still want to save to config file)

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		nSteps = state['nSteps']
		s = nSteps / self._max_steps
		value = 1-1/math.exp(math.log(0.5)*s**2)
		done = False
		if nSteps >= self._max_steps:
			done = True
		if done and self.terminate:
			value = -10
			state['termination_reason'] = 'steps'
			state['termination_result'] = 'failure'
		return value, done and self.terminate

	# add to max steps based on goal distance
	def reset(self, state):
		if self.update_steps:
			_drone_position = np.array(state['drone_position'])
			_goal_position = np.array(state['goal_position'])
			distance_vector = _goal_position - _drone_position
			distance = np.linalg.norm(distance_vector)
			self._max_steps = self.max_steps + int(self.step_ratio*distance)