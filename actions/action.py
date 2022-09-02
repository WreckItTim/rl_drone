# abstract class used to handle an individual action
from component import Component

class Action(Component):

	# contstructor
	def __init__(self):
		pass
		
	# when using the debug controller
	def debug(self):
		self.act()

	def connect(self):
		super().connect()