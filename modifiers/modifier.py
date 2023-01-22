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
# define activate(state) from child class
# by default, Modifier parent will activate for any parent method
# optionally, override any parent methods for method-specific modifications
class Modifier(Component):

	def __init__(self,
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  order, # modify 'pre' or 'post'?
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = False, # will activate on first call otherwise only if % is not 0
			  ):
		# modify base method with this one
		base_method = getattr(self._base, parent_method)
		modification = getattr(self, parent_method)
		setattr(self._base, parent_method, _modify(base_method, modification, order))
		self.connect_priority = -999 # connects after everything else

	# increments counter and checks if we activate based on counter
	def check_counter(self):
		self.counter += 1
		return (self.counter == 1 and self.activate_on_first or
		self.counter % self.frequency == 0)

	# define this from child - this is whatever the modifier does
	def activate(self, state=None):
		raise NotImplementedError

	# takes a step in an episode
	def step(self, state=None):
		self.activate(state)

	# resets and end of episode to prepare for next
	def reset(self, state=None):
		self.activate(state)

	# use to reset learning loop attributes
	def reset_learning(self, state=None):
		self.activate(state)
	
	# kill connection, clean up as needed
	def disconnect(self, state=None):
		self.activate(state)

	# write any vars to file
	def save(self, write_folder, state=None):
		# add write_folder to state if you need to use it then del from state
		self.activate(state)

	# does whatever to check whatever (used for debugging mode)
	def debug(self, state=None):
		self.activate(state)
		
	# establish connection to be used in episode - connects all components to eachother and calls child connect() for anything else needed
	# WARNING: if you overwrite this make sure to call super()
	def connect(self, state=None):
		super().connect(state)
		if self.parent_method == 'connect':
			self.activate(state)