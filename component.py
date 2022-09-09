import utils
import numpy as np
from time import time
from sys import getsizeof
import inspect
from functools import wraps, partialmethod
from configuration import Configuration

# README - for all new classes that you make (I suggest everyone read this to understand how the repo works anyways):
# all classes are children of this Component class 
# this uses overlapping logic for serialization, connecting, running, logging, conflict management...
# after you know how this works you can copy and paste new child classes and have them integrated into the repo, working in seconds
# 1. if you are making a new parent class named {parent}: ---- note that it's probably easier to just make a new "Other" child class ----
	# a. add a folder named '{parent}s' and within it add a python file named '{parent}.py'
	# b. create a new class named '{Parent}' in the {parent}.py file
	# c. have the {Parent} class inherit Component, such as 'class {Parent}(Component):' - note the capital P
	# d. optionally, define any class methods as wanted from the Component class, seen below
	# e. define a {Parent}.connect() method that calls super, such as 'def connect(self): super().connect()'
# 2. if you are making a new child class named {child} from the parent class named {parent}:
	# a. add a python file named '{child}.py' inside the existing folder named '{parent}s'
	# b. create a new class named '{Child}' in the {child}.py file
	# c. have the {Child} class inherit {Parent}, such as 'class {Child}({Parent}):' - note the capital C
	# d. decorate {Child}.__init__() with @_init_wrapper - see definition below
	# e. define any necesary (abstract) class methods as declared in {Parent}
	# f. optionally, define any class methods as wanted from the Component class, seen below
	# g. if you redefine {Child}.connect() make sure to call super, such as 'def connect(self): super().connect() ...'
# 3. if you want to have a Component class from {MyComponent} as a member {member} in another Class {MyClass}:
	# NOTE, so that order of component creation does not matter, the following is done:
	# NOTE, all global Component instances for a configuration are first created before connecting Component members
	# NOTE, otherwise order can create conflicts, the following also allows for automatic serialization
	# a. define an argument named {member}_component in the {MyClass}.__init__() method, such as {MyClass}.__init__(..., {member}_component)
		# let a Class instance belonging to {MyClass} be {class_instance}
		# let a Component instance belonging to {MyComponent} be {component_instance}
		# such that we want: {class_instance}._{member}={component_instance} - note that member will be 'private' (necesary for serialization)
		# during creation of a {component_instance}, you can pass a string argument {component_name} for {component_instance}
			# , such as {component_instance}={MyComponent}.__init__(..., name={member_name})
		# {component_instance} will set the unique string ID upon creation, such as {member_instance}._name={member_name}
			# , and save it to the configuraition's component_list.
		# This way you can pseudo-define a {class_instance}._{member}={component_instance} before {component_instance} is even created!
		# upon creation of {class_instance}, this is done: {class_instance}.{member}_component = {component_name}
		# after creating all components, the configuration object will call connect() for all components and set {member}
			# , such as {class_instance}._{member}=get_component({class_instance}.{member}_component) 
		# the argument {member}_component passed into {MyClass}.__init__() can also be the exact {component_instance} if already created
		# NOTE, any {member}_component arguments to {MyClass}.__init__() will automatically set a private class member after connect() is called
			# , such as {class_instance}._{member}={component_instance}
		# NOTE, built-in, automatic deserialization leverages the above methodology as well
		# now, if you want a component to connect first or last, set {class_instance}.connect_priority={x}. I typicaly set this from {MyClass}.__init__()
		# priority will load lowest-to-highest positive {x}, then all 0-priority (default) components, then highest-to-lowest negative {x}
		# priority load example: (1) (2) (3) (0) (0) (0) (0) (-1) (-2)
		# ties in priority will run in order of instance creation
		# similarily you can set {class_instance}.disconnect_priority={y}
# 4. if you want to have a list of Component members {member} from various Component classes, in another Class {MyClass}:
	# a. add an argument named {member}_components to the {MyClass}.__init__() method - note the 's' in components
		# defining a list of components follows the same logic as in section 3 above...
		# the {member}_components argument to {MyClass}.__init__() is a mixed list of {component_instance} and {component_name} variables
		# let {component_instances} be the hypothetical list of all the desired {component_instance} variables
		# all {member}_components arguments to {MyClass}.__init__() will automatically set a private class member after connect() is called
			# , such as self._{member}={component_instances}
# IF you followed the above steps, then your new component class will:
	# a. be serializable (for configuration files)
	# b. will avoid conflicts so the order of creation does not matter
	# c. can be time/memory benchmarked
	# d. can be used with other components
	# e. can be used in the same manner as any parent-base class, for example you can make a new reward or action or sensor or whatever class 
	
# times all funciton calls and saves to library (microseconds)
def _timer_wrapper(configuration, method):
	def _wrapper(*args, **kwargs):
		module_name = method.__module__
		t1 = time()
		method_output = method(*args, **kwargs)
		t2 = time()
		delta_t = (t2 - t1) * 1000000
		entry_name = module_name + '.' + method.__name__
		configuration.log_benchmark('time', entry_name, delta_t)
		return method_output
	return _wrapper

# sets intialization values
def _init_wrapper(init_method):
	sig = inspect.signature(init_method)

	@wraps(init_method)
	def _wrapper(self, *args, **kwargs):

		# GET WRAPPER VARS
		configuration = kwargs['configuration']
		del kwargs['configuration']
		arg_name = kwargs['name']
		del kwargs['name']
		connect_priority = kwargs['connect_priority']
		del kwargs['connect_priority']
		disconnect_priority = kwargs['disconnect_priority']
		del kwargs['disconnect_priority']

		# GET CONFIGURATION OBJECT
		if configuration is None:
			configuration = Configuration.get_active()
		self._configuration = configuration

		# UPDATE ALL ARGS
		bound = sig.bind(self, *args, **kwargs)
		bound.apply_defaults()
		all_args = bound.arguments

		# SET ALL PUBLIC PASSED IN ARGUMENTS AS CLASS ATTRIBUTES
		# a _component argument can be read as a string name or as an object (used in serialization)
		# component(s) arguments save all as strings now and will set when connect is called (so that order of creation does not matter)
		self._connect_components_list = [] # list of pairs (member_name, component_names)
		self._connect_component_list =  [] # list of pairs (member_name, component_name)
		for key in all_args:
			# skip over private arguments
			if key[0] == '_' or key == 'self':
				continue
			# lists of components
			elif '_components' in key:
				values = all_args[key]
				member_name = '_' + key.replace('_components', '')
				if values is None:
					component_names = None
				else:
					component_names = []
					for value in values:
						if isinstance(value, str):
							component_name = value
						elif isinstance(value, Component):
							component_name = value._name
						else:
							utils.error('passed in argument as _component in _components list, but argument is not str or component type')
						component_names.append(component_name)
				self._connect_components_list.append((member_name, component_names))
				setattr(self, key, component_names)
			# individual components
			elif '_component' in key:
				value = all_args[key]
				member_name = '_' + key.replace('_component', '')
				if value is None:
					component_name = None
				elif isinstance(value, str):
					component_name = value
				elif isinstance(value, Component):
					component_name = value._name
				else:
					utils.error('passed in argument as _component but is not str or component type')
				self._connect_component_list.append((member_name, component_name))
				setattr(self, key, component_name)
			# set all (other) public arguments
			else:
				setattr(self, key, all_args[key])

		# CALL BASE INIT
		self._add_timers = True # change in base init method to false to not add
		self._set_name = True # change in base init method to false to not add
		self._add_to_configuration = True # change in base init method to false to not add
		init_method(self, *args, **kwargs)

		# SET PIRORITIES for load orders, to default of 0 if not set yet or not passed in as argument
		if connect_priority is None and getattr(self, "connect_priority", None) is None:
			self.connect_priority = 0 
		if disconnect_priority is None and getattr(self, "disconnect_priority", None) is None:
			self.disconnect_priority = 0

		# ADD TIMER TO EACH PUBLIC CLASS METHOD
		if self._add_timers:
			for method in dir(self):
				if callable(getattr(self, method)) and method[0] != '_':
					setattr(self, method, _timer_wrapper(configuration, getattr(self, method)))

		# SET UNIQUE NAME
		if self._set_name:
			clone_id = 1
			if arg_name is None:
				partial_name = f'{self._child().__name__}'
				try_name = f'{partial_name}__{clone_id}' 
			else:
				partial_name = arg_name
				try_name = partial_name
			clone_id = 2
			while try_name in configuration.components:
				try_name = f'{partial_name}__{clone_id}'
				clone_id = clone_id + 1
			self._name = try_name

		# ADD TO LIST OF COMPONENTS
		if self._add_to_configuration:
			configuration.add_component(self)
		
	return partialmethod(_wrapper, 
					  configuration = None, 
					  name = None, 
					  connect_priority = None, 
					  disconnect_priority = None,
					  )

# the component class itself
class Component():
	# decorate all child sub-Component classes __init__() like this (wihtout the comment):
	#@_init_wrapper
	def __init__(self):
		pass


	# CLASS METHODS - ovewrite as needed from child

	# estimate memory used by this component (in bytes)
	def __sizeof__(self):
		total_memory = 0
		variables = vars(self)
		for key in variables:
			variable = variables[key]
			if callable(variable):
				continue
			if isinstance(variable, Component):
				total_memory += 8
			else: 
				total_memory += getsizeof(variable)
		return total_memory

	# establish connection to be used in episode - connects all components to eachother and calls child connect() for anything else needed
	# if you overwrite this make sure to call super()
	def connect(self):
		for components_pair in self._connect_components_list:
			member_name, component_names = components_pair
			if component_names is None:
				components = None
			else:
				components = []
				for component_name in component_names:
					component = self._configuration.get_component(component_name)
					components.append(component)
			setattr(self, member_name, components)
		for component_pair in self._connect_component_list:
			member_name, component_name = component_pair
			if component_name is None:
				component = None
			else:
				component = self._configuration.get_component(component_name)
			setattr(self, member_name, component)

	# kill connection, clean up as needed
	def disconnect(self):
		pass

	# does whatever to check whatever (used for debugging mode)
	def debug(self):
		pass

	# resets and end of episode to prepare for next
	def reset(self):
		pass

	# stops component - if error is reached
	def stop(self):
		self.disconnect()


	# HELPER METHODS

	# throws error if not correct child type
	def check(self, child_type):
		ok = isinstance(self, child_type) 
		if not ok:
			utils.error(f'Can not handle child type of {self._child().__name__}, requires {child_type.__name__}')

	# return parent type
	def _parent(self):
		return type(self).__bases__[0]
		
	# return child type
	def _child(self):
		return type(self)

	# turns component into a dictionary/json format for serialization - auto gets class _name and self-serializable parameters
	def _to_json(self):
		component_arguments = {'type':f'{self._parent().__name__}.{self._child().__name__}'}
		variables = vars(self)
		for key in variables:
			value = variables[key]
			if callable(value):
				continue
			if key[0] == '_':
				continue
			if key == 'observation_space' or key == 'action_space':
				continue
			if (key == 'connect_priority' or key == 'disconnect_priority') and value == 0:
				continue
			if type(value) is np.ndarray:
				value = value.tolist()
			component_arguments[key] = value
		return component_arguments

	# checks json serialization for equality
	def __eq__(self, other):
		return self._to_json() == other._to_json()

	# deserializes a dictionary to a component
	@staticmethod
	def deserialize(component_name, component_arguments):
		# get parent and child types
		types = component_arguments['type'].split('.')
		parent_type, child_type = types[0], types[1]
		import_component = parent_type.lower() + 's.' + child_type.lower()
		child_module = __import__(import_component, fromlist=[child_type])
		child_class = getattr(child_module, child_type)
		# create a child object
		component_arguments_copy = component_arguments.copy()
		del component_arguments_copy['type']
		if 'name' not in component_arguments_copy:
			component_arguments_copy['name'] = component_name
		component = child_class(**component_arguments_copy)
		return component

from controllers.controller import Controller