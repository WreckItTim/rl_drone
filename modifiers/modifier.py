from component import Component
import inspect

# wrapper method to modify a base method
def _modify(base_method, modification, order):
	def _wrapper(*args, **kwargs):
		if order == 'pre':
			modification(*args, **kwargs)
		method_output = base_method(*args, **kwargs)
		if order == 'post':
			modification(*args, **kwargs)
		return method_output
	return _wrapper

# abstract class used to handle modifiers
# apply to any component and parent method
class Modifier(Component):

	def __init__(self,
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  order, # modify 'pre' or 'post'?
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
	):
		# modify base method with this one
		base_method = getattr(self._base, parent_method)
		modification = getattr(self, parent_method)
		setattr(self._base, parent_method, _modify(base_method, modification, order))
		self.connect_priority = -999 # connects after everything else