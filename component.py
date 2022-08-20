import numpy as np
from time import time
import inspect
from sys import getsizeof
from utils import error
from functools import wraps

# all my classes are children of this Component class 
# I use alot of overlapping logic for serialization, connecting, running, logging...

# serializes a list of components into a configuration file
def serialize_components(components):
	configuration = {}
	for component in components:
		configuration[component._name] = component._to_json()
	return configuration

# deserializes a configuration file into a list of components
def deserialize_components(configuration):
	components = []
	for component_name in configuration:
		component_arguments = configuration[component_name]
		component = deserialize_component(component_name, component_arguments)
		components.append(component)
	return components

# deserializes a json file to a component
def deserialize_component(component_name, component_arguments):
	# get parent and child types
	types = component_arguments['type'].split('.')
	parent_type, child_type = types[0], types[1]
	import_component = parent_type.lower() + 's.' + child_type.lower()
	child_module = __import__(import_component, fromlist=[child_type])
	child_class = getattr(child_module, child_type)
	# create a child object
	del component_arguments['type']
	component_arguments['_name'] = component_name
	component = child_class(**component_arguments)
	return component

# keeps track of global components
component_list = {}
def get_component(_name, is_type=None):
	if _name not in component_list:
		print('component named', _name, 'does not exist')
		return None
	component = component_list[_name]
	if is_type is not None:
		component.check(is_type)
	return component

# returns a list of all components created
def get_all_components():
	components = []
	for component_name in component_list:
		components.append(get_component(component_name))
	return components

# logs time and memory benchmarks
benchmarks = {'time':{'units':'microseconds'}, 'memory':{'units':'kilobytes'}, 'event':{}}
def log_entry(master_key, key, value):
	if key not in benchmarks[master_key]:
		benchmarks[master_key][key] = [value]
	else:
		benchmarks[master_key][key].append(value)

# fetches memory stored in this object, as mega bytes
def log_memory(component):
	nBytes = getsizeof(component)
	nKiloBytes = nBytes * 0.000977
	log_entry('memory', component._name, nKiloBytes)

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
		# update arguments with defaults and make one giant dictionary of all args
		bound = sig.bind(self, *args, **kwargs)
		bound.apply_defaults()
		all_args = bound.arguments
        del all_args['name']

		# SET ALL PUBLIC PASSED IN ARGUMENTS AS CLASS ATTRIBUTES
		for key in all_args:
			# convert public list of component _names to private list of components
			# skip over private arguments
			if key[0] == '_' or key == 'self':
				continue
			# parse and set list of components
			elif 'components' in key:
				values = all_args[key]
				components = []
				for value in values:
					if isinstance(value, str):
						component = get_component(value)
					elif isinstance(value, Component):
						component = value
					else:
						error('passed in argument as _component in _components list, but argument is not str or component type')
					components.append(component)
				setattr(self, '_' + key.replace('_components', 's'), components)
			# parse and set component
			elif 'component' in key:
				value = all_args[key]
				if isinstance(value, str):
					component = get_component(value)
				elif isinstance(value, Component):
					component = value
				else:
					error('passed in argument as _component but is not str or component type')
				setattr(self, '_' + key.replace('_component', ''), component)
			# set all public arguments
			else:
				setattr(self, key, all_args[key])

		# ADD TIMER TO EACH PUBLIC CLASS METHOD
		for method in dir(self):
			if callable(getattr(self, method)) and method[0] != '_':
				setattr(self, method, _timer_wrapper(getattr(self, method)))

		# SET UNIQUE _name 
		clone_id = 1
		if '_name' not in all_args:
			_name = f'{self._child().__name__}'
			try_name = f'{_name}__{clone_id}' 
		else:
			_name = all_args['_name']
			try_name = _name
		clone_id = 2
		while try_name in component_list:
			try_name = f'{_name}__{clone_id}'
			clone_id = clone_id + 1
		self._name = try_name

		# CALL BASE INIT
		init_method(self, *args, **kwargs)

		# ADD TO LIST OF COMPONENTS
		component_list[self._name] = self
		
    return partial(_wrapper, name=None)

# the component class itself
class Component():
	# WRAP ALL __INIT__ WITH THIS or call super() to not set class attributes automatically
    #@_init_wrapper
	def __init__(self):
	
		# ADD TIMER TO EACH PUBLIC CLASS METHOD
		for method in dir(self):
			if callable(getattr(self, method)) and method[0] != '_':
				setattr(self, method, _timer_wrapper(getattr(self, method)))

		# SET UNIQUE _name 
		clone_id = 1
		if _name is None:
			_name = f'{self._child().__name__}'
			try_name = f'{_name}__{clone_id}' 
		else:
			try_name = _name
		clone_id = 2
		while try_name in component_list:
			try_name = f'{_name}__{clone_id}'
			clone_id = clone_id + 1
		self._name = try_name

	# gets bytes of all components
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

	# establish connection to be used in episdoe
	def connect(self):
		pass

	# activates whatever to do whatever
	def activate(self):
		pass

	# kill connection, clean up as needed
	def disconnect(self):
		pass

	# resets and end of episode to prepare for next
	def reset(self):
		pass

	# checks to insure component is running properly
	def test(self):
		pass

	# stops component - if error is reached
	def stop(self):
		self.disconnect()

	# called every step in RL, and updates state dictionary
	def step(self, state):
		pass
	
	# throws error if not correct child type
	def check(self, child_type):
		ok = isinstance(self, child_type) 
		if not ok:
			error(f'Can not handle child type of {self._child().__name__}, requires {child_type.__name__}')

	# return parent type
	def _parent(self):
		return type(self).__bases__[0]
		
	# return child type
	def _child(self):
		return type(self)

	# turns component into a dictionary/json format - auto gets class _name and self-serializable parameters
	def _to_json(self):
		component_arguments = {'type':f'{self._parent().__name__}.{self._child().__name__}'}
		variables = vars(self)
		for key in variables:
			variable = variables[key]
			if callable(variable):
				continue
			if key[0] == '_':
				continue
			if key == 'observation_space' or key == 'action_space':
				continue
			if type(variable) is np.ndarray:
				variable = variable.tolist()
			component_arguments[key] = variable
		return component_arguments
      
	# checks json serialization for equality
	def __eq__(self, other):
		if self._to_json()==other._to_json():
			return True
		return False