from component import Component

# abstract class used to handle actions
class Actor(Component):
	# contstructor
	def __init__(self,
			  actions_components=[], 
			  _state=None,
			  ):
		raise NotImplementedError
	
	# resets and end of episode to prepare for next
	def reset(self, state=None):
		for action in self._actions:
			action.reset(state)
			
	# returns action space for this actor
	def get_space(self):
		raise NotImplementedError
