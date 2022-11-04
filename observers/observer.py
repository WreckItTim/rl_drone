# abstract class used to handle observations to input into rl algo
from component import Component
from os.path import exists
from os import makedirs

class Observer(Component):
	# constructor
	def __init__(self):
		pass

	# when using the debug controller
	def debug(self):
		self.observe().display()

	def connect(self):
		super().connect()

	# returns observation transcribed for input into RL model
	def observe(self):
		raise NotImplementedError

	def reset(self):
		pass

	# returns observation space for this observer
	def get_space(self):
		raise NotImplementedError