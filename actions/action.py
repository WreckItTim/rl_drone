from component import Component

# abstract class used to handle an individual action
class Action(Component):

	# contstructor
	def __init__(self, _state=None):
		raise NotImplementedError
	
	# establish connection to be used in episode - connects all components to eachother and calls child connect() for anything else needed
	# WARNING: if you overwrite this make sure to call super()
	def connect(self, state=None):
		super().connect()