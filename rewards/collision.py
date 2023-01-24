from rewards.reward import Reward
from component import _init_wrapper

# penalizes/terminates when colliding with objects
class Avoid(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, 
				 drone_component,
				 terminate=True, # =True will terminate episodes on collision
				 ):
		super().__init__()

	# -1 for a collision, +1 for dodging collision
	def step(self, state):
		if 'has_collided' not in state:
			state['has_collided'] = self._drone.check_collision()
		has_collided = state['has_collided']
		value = 0
		done = False
		if has_collided:
			value = -10
			done = True
		if done and self.terminate:
			state['termination_reason'] = 'collision'
			state['termination_result'] = 'failure'
		return value, done and self.terminate