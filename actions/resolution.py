from actions.action import Action
from component import _init_wrapper
import math

# coutinuouts output will scale resolution of input sensors
# sensors should start at maximum scale
class Resolution(Action):
	# constructor takes a list of components to scale down resolution
	@_init_wrapper
	def __init__(self, 
				scales_components, # list of components to call scale_down() on
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_space = -1, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				max_level = 3, # levels of resolution scale down by (do not include zero)
					# for example, if want to scale down by zero or one level then levels=1
				adjust_for_yaw = False,
				active = True, # False will just return default behavior
			):
		pass
		
	# move at input rate
	def step(self, state, execute=True):
		if not self.active:
			level = self.max_level
		else:
			rl_output = state['rl_output'][self._idx]
			level = max(0, min(int((self.max_level+1) * rl_output), self.max_level))
		self._level = level # give information access compos connected to this
		# tell each resolution-based component to scale
		for scale in self._scales:
			scale.scale_to(level)
		return {self._name:level}