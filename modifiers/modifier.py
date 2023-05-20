from component import Component
from environments.environment import Environment
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
			  on_evaluate = True, # toggle to run modifier on evaluation environ
			  on_train = True, # toggle to run modifier on train environ
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0 , # keepts track of number of calls to parent method
			  ):
		self.connect_priority = -999 # connects after everything else

	# increments counter and checks if we activate
	def check_counter(self, state=None):
		if isinstance(self._base, Environment) or isinstance(self._base, Environment):
			if state is None:
				state = {}
			state['is_evaluation_env'] = self._base.is_evaluation_env
		if state is not None and 'is_evaluation_env' in state:
			# check if we do not activate on eval environ
			if state['is_evaluation_env']:
				if not self.on_evaluate:
					return False
			# check if we do not activate on train environ
			else:
				if not self.on_train:
					return False
		# we now know that we are in proper environment...
		self.counter += 1
		# check if we are on frequency
		if self.counter % self.frequency == 0:
			return True
		return False

	# define this from child - this is whatever the modifier does
	def activate(self, state=None):
		raise NotImplementedError

	# takes a step in an episode
	def step(self, state=None):
		if self.parent_method == 'step':
			self.activate(state)
			
	# called at the beginning of an episode to prepare for next
	def start(self, state=None):
		if self.parent_method == 'start':
			self.activate(state)

	# called at the end of an episode to clean up
	def end(self, state=None):
		if self.parent_method == 'end':
			self.activate(state)

	# use to reset learning loop attributes
	def reset_learning(self, state=None):
		if self.parent_method == 'reset_learning':
			self.activate(state)
	
	# kill connection, clean up as needed
	def disconnect(self, state=None):
		if self.parent_method == 'disconnect':
			self.activate(state)

	# write any vars to file
	def save(self, state=None):
		# add write_folder to state if you need to use it then del from state
		if self.parent_method == 'save':
			self.activate(state)

	# does whatever to check whatever (used for debugging mode)
	def debug(self, state=None):
		if self.parent_method == 'debug':
			self.activate(state)
		
	# establish connection to be used in episode - connects all components to eachother and calls child connect() for anything else needed
	# WARNING: if you overwrite this make sure to call super()
	def connect(self, state=None):
		super().connect(state)
		# modify base method with this one
		base_method = getattr(self._base, self.parent_method)
		modification = getattr(self, self.parent_method)
		setattr(self._base, self.parent_method, _modify(base_method, modification, self.order))
		if self.parent_method == 'connect':
			self.activate(state)
