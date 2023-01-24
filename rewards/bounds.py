from rewards.reward import Reward
from component import _init_wrapper

# rewards/terminatess based on x,y,z boundaries
class Bounds(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, 
                 drone_component, 
                 x_bounds,
                 y_bounds,
                 z_bounds,
				 terminate=False, # =True will terminate episodes when oob
	):
		super().__init__()

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		_drone_position = self._drone.get_position()
		x = _drone_position[0]
		y = _drone_position[1]
		z = _drone_position[2]
		value = 0
		done = False
		if x < self.x_bounds[0] or x > self.x_bounds[1]:
			value = -10
			done = True
		if y < self.y_bounds[0] or y > self.y_bounds[1]:
			value = -10
			done = True
		if z < self.z_bounds[0] or y > self.z_bounds[1]:
			value = -10
			done = True
		if done and self.terminate:
			state['termination_reason'] = 'bounds'
			state['termination_result'] = 'failure'
		return value, done and self.terminate