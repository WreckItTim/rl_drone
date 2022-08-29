import numpy as np
from time import time
from sys import getsizeof
import utils
import inspect
from functools import wraps, partialmethod

# README - for all new classes that you make (I suggest everyone read this to understand how the repo works anyways):
# all classes are children of this Component class 
# this uses overlapping logic for serialization, connecting, running, logging, conflict management...
# after you know how this works you can copy and paste new child classes and have them integrated into the repo, working in seconds with little extra coding required!
# 1. if you are making a new parent class named {parent}: ---- note that it's probably easier to just make a new "Other" child class ----
	# a. add a folder named '{parent}s' and within it add a python file named '{parent}.py'
	# b. create a new class named '{Parent}' in the {parent}.py file
	# c. have the {Parent} class inherit Component, such as 'class {Parent}(Component):' - note the capital P
	# d. optionally, define any class methods as declared in the Component class, seen below
	# e. define a {Parent}.connect() method that calls super, such as 'def connect(self): super().connect()'
# 2. if you are making a new child class named {child} from the parent class named {parent}:
	# a. add a python file named '{child}.py' inside the existing folder named '{parent}s'
	# b. create a new class named '{Child}' in the {child}.py file
	# c. have the {Child} class inherit {Parent}, such as 'class {Child}({Parent}):' - note the capital C
	# d. decorate {Child}.__init__() with @_init_wrapper - see definition below
	# e. define any necesary (abstract) class methods as declared in {Parent}
	# f. optionally, define any class methods as declared in the Component class, seen below
	# g. if you redefine {Child}.connect() make sure to call super, such as 'def connect(self): super().connect() ...'
# 3. if you want to have a Component class named {Component} as a member named {member} in another Class named {Class}:
	# a. define an argument named {member}_component in the {Class}.__init__() method
		# NOTE, so that order does not matter, for component creation, the following is done:
		# (otherwise order can create conflicts) (this also allows for automatic serialization)
		# let a global Class instance belonging to {Class} be {class_instance}
		# let another global Component instance belonging to {Component} be {member_instance}
		# such that we want: {class_instance}._{member}={member_instance} - note that member will be private (necesary for serialization)
		# all global Component instances, needed for a given configuration, are first created before any Component members are properly set
		# you can pass a string argument {member_name} when creating {member_instance}, such as {member_instance}={Component}.__init__(..., name={member_name})
		# {member_instance} will set the unique string ID upon creation, such as {member_instance}._name={member_name}, and save it to the global component_list.
		# This way you can define a {class_instance}.{member} before {member_instance} is even created!
		# upon creation of {class_instance}, this is done: {class_instance}.{member}_component = {member_name}
		# all class Component members are properly set during Component.connect(), such as {class_instance}._{member}=get_component({class_instance}.{member}_component) 
		# the argument {member}_component passed into {Class}.__init__() can be the exact {member_instance} if already created, or {member_name} if otherwise not yet created
		# all {member}_component arguments to {Class}.__init__() will automatically set a private class member after connect() is called, such as self._{member}={member_instance}
		# deserialization, reading from a configuation file, also leverages the above methodology so that your component classes can automatically be serialized/deserialized
		# if you want a component to connect first or last, set {class_instance}.connect_priority={x}. I typicaly set this from {Class}.__init__()
		# priority will load lowest-to-highest positive {x} before all 0-priority (default) components, and highest-to-lowest negative {x} after all 0-priority (default) components
		# priority load example: (1) (2) (3) (0) (0) (0) (0) (-1) (-2)
		# ties in priority will run in order of instance creation
		# similarily you can set {class_instance}.disconnect_priority={y}
# 4. if you want to have a list of Component members named {members} in another Class named {Class}:
	# a. add an argument named {member}_components to the {Class}.__init__() method - note the s at the end
		# read secton 3, defining a list of components follows the same logic
		# the {member}_components argument to {Class}.__init__() is a mixed list of {member_instance} and {member_name} like variables
		# let {member_instances} be the hypothetical list of all the desired {member_instance} variables
		# all {member}_components arguments to {Class}.__init__() will automatically set a private class member after connect() is called, such as self._{member}={member_instances}
# IF you followed steps 1-5 correctly, than your new component class will:
	# a. be serializable (for configuration files)
	# b. will avoid conflicts so the order of creation does not matter
	# c. can be time/memory benchmarked
	# d. can be used with other components
	# e. can be used in the same manner as any parent-base class, for example you can make a new reward or action or sensor or whatever
	
# times all funciton calls and saves to library (microseconds)
def _timer_wrapper(method):
	def _wrapper(*args, **kwargs):
		module_name = method.__module__
		t1 = time()
		method_output = method(*args, **kwargs)
		t2 = time()
		delta_t = (t2 - t1) * 1000000
		entry_name = module_name + '.' + method.__name__
		log_entry('time', entry_name, delta_t)
		return method_output
	return _wrapper

# sets intialization values
def _init_wrapper(init_method):
	sig = inspect.signature(init_method)

	@wraps(init_method)
	def _wrapper(self, *args, **kwargs):

		# SET UNIQUE NAME 
		clone_id = 1
		arg_name = kwargs['name']
		del kwargs['name']
		if arg_name is None:
			partial_name = f'{self._child().__name__}'
			try_name = f'{partial_name}__{clone_id}' 
		else:
			partial_name = arg_name
			try_name = partial_name
		clone_id = 2
		while try_name in component_list:
			try_name = f'{partial_name}__{clone_id}'
			clone_id = clone_id + 1
		self._name = try_name

		# TEMP SAVE PIRORITIES for load orders
		connect_priority = kwargs['connect_priority']
		del kwargs['connect_priority']
		disconnect_priority = kwargs['disconnect_priority']
		del kwargs['disconnect_priority']

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
				if isinstance(value, str):
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
		self._add_to_list = True # change in base init method to false to not add
		self._add_timers = True # change in base init method to false to not add
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
					setattr(self, method, _timer_wrapper(getattr(self, method)))

		# ADD TO LIST OF COMPONENTS
		if self._add_to_list:
			component_list[self._name] = self
		
	return partialmethod(_wrapper, name=None, connect_priority=None, disconnect_priority=None)

# serializes a list of components into a configuration file
def serialize_configuration(controller, components, timestamp, repo_version):
	configuration = {
		'timestamp':timestamp,
		'repo_version':repo_version,
		'controller':controller._to_json(),
	}
	for component in components:
		configuration[component._name] = component._to_json()
	return configuration

# deserializes a configuration file into a list of components
def deserialize_configuration(configuration):
	timestamp = configuration['timestamp']
	repo_version = configuration['repo_version']
	controller_arguments = configuration['controller']
	components = []
	for component_name in configuration:
		if component_name in ['controller', 'timestamp', 'repo_version']:
			continue
		component_arguments = configuration[component_name]
		component = deserialize_component(component_name, component_arguments)
		components.append(component)
	controller = deserialize_component('controller', controller_arguments)
	return controller, components, timestamp, repo_version

# deserializes a dictionary to a component
def deserialize_component(component_name, component_arguments):
	# get parent and child types
	types = component_arguments['type'].split('.')
	parent_type, child_type = types[0], types[1]
	import_component = parent_type.lower() + 's.' + child_type.lower()
	child_module = __import__(import_component, fromlist=[child_type])
	child_class = getattr(child_module, child_type)
	# create a child object
	component_arguments_copy = component_arguments.copy()
	del component_arguments_copy['type']
	component_arguments_copy['name'] = component_name
	component = child_class(**component_arguments_copy)
	return component

# keeps track of global components
component_list = {}
def get_component(component_name, is_type=None):
	if component_name not in component_list:
		print('component named', component_name, 'does not exist')
		return None
	component = component_list[component_name]
	if is_type is not None:
		component.check(is_type)
	return component

# returns a list of all global components created
def get_all_components():
	components = []
	for component_name in component_list:
		components.append(get_component(component_name))
	return components

# connects in order of connect_priority (positives low-to-high, default zeros, negatives high-to-low)
def connect_components(components):
	# get priorities
	priorities = []
	component_dic = {}
	for idx, component in enumerate(components):
		priority = component.connect_priority
		if priority not in priorities:
			priorities.append(priority)
			component_dic[priority] = []
		component_dic[priority].append(component)
	priorities.sort()
	# positives first
	start_positive_index = len(priorities)
	for index, priority in enumerate(priorities):
		if priority > 0:
			start_positive_index = index
			break
	for index in range(start_positive_index, len(priorities)):
		for component in component_dic[priorities[index]]:
			component.connect()
	# default zeroes
	if 0 in component_dic:
		for component in component_dic[0]:
			component.connect()
	# negatives last
	start_negative_index = 0
	for index, priority in enumerate(reversed(priorities)):
		if priority < 0:
			start_negative_index = len(priorities)-index-1
			break
	for index in range(start_negative_index, -1, -1):
		for component in component_dic[priorities[index]]:
			component.connect()

# diusconnects in order of disconnect_priority (positives low-to-high, default zeros, negatives high-to-low)
def disconnect_components(components):
	# get priorities
	priorities = []
	component_dic = {}
	for idx, component in enumerate(components):
		priority = component.disconnect_priority
		if priority not in priorities:
			priorities.append(priority)
			component_dic[priority] = []
		component_dic[priority].append(component)
	priorities.sort()
	# positives first
	start_positive_index = len(priorities)
	for index, priority in enumerate(priorities):
		if priority > 0:
			start_positive_index = index
			break
	for index in range(start_positive_index, len(priorities)):
		for component in component_dic[priorities[index]]:
			component.disconnect()
	# default zeroes
	if 0 in component_dic:
		for component in component_dic[0]:
			component.disconnect()
	# negatives last
	start_negative_index = 0
	for index, priority in enumerate(reversed(priorities)):
		if priority < 0:
			start_negative_index = len(priorities)-index-1
			break
	for index in range(start_negative_index, -1, -1):
		for component in component_dic[priorities[index]]:
			component.disconnect()

# logs global time and memory benchmarks, to be written at end (or at intervals if using Other.BenchmarkWriter)
benchmarks = {'time':{'units':'microseconds'}, 'memory':{'units':'kilobytes'}}
def log_entry(master_key, key, value):
	if key not in benchmarks[master_key]:
		benchmarks[master_key][key] = [value]
	else:
		benchmarks[master_key][key].append(value)

# fetches memory stored in this object, in mega bytes
def log_memory(component):
	nBytes = getsizeof(component) # self.__sizeof__()
	nKiloBytes = nBytes * 0.000977
	log_entry('memory', component._name, nKiloBytes)

# benchmarks all components and writes to file
def benchmark_components(components, write_path=None):
	for component in components:
		log_memory(component)
	if write_path is None:
		write_path = utils.get_global_parameter('write_folder') + 'benchmarks.json'
	utils.write_json(benchmarks, write_path)

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
			components = []
			for component_name in component_names:
				component = get_component(component_name)
				components.append(component)
			setattr(self, member_name, components)
		for component_pair in self._connect_component_list:
			member_name, component_name = component_pair
			component = get_component(component_name)
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