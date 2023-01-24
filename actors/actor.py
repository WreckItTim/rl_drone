from component import Component

# abstract class used to handle actions
class Actor(Component):
	# contstructor
	def __init__(self,
			  ):
		self.connect_priority = 1 # before environ to get_space
	
	# resets and end of episode to prepare for next
	def reset(self, state=None):
		for action in self._actions:
			action.reset(state)
			
	# returns action space for this actor
	def get_space(self):
		raise NotImplementedError
