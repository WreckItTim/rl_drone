# abstract class used to handle rewards in an RL enivornment
from component import Component

class Terminator(Component):
	# constructor
	def __init__(self):
		pass

	# evaluates if termination criteria is met, ruturns done=True if so otherwise done=False
	def terminate(self, state):
		raise NotImplementedError

	# when using the debug controller
	def debug(self):
		return self.terminate({})

	def connect(self):
		super().connect()