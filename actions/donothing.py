from actions.action import Action
from component import _init_wrapper
import math
import rl_utils as utils

# literally does nothing - this is so there is an analogue between continuous with true-zero and discrete
class DoNothing(Action):
	@_init_wrapper
	def __init__(self,
			  ):
		pass