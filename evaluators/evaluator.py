# abstract class used to handle evaluators
from component import Component

class Evaluator(Component):
	# constructor
	def __init__(self):
		pass
	def connect(self, state=None):
		super().connect(state)