import numpy as np
from time import time
import inspect
from sys import getsizeof

# all my classes are children of this Component class 
# I use alot of overlapping logic for serialization, connecting, running, logging...

# serializes a list of components into a parameter dictionary
def serialize_components(components):
	parameters = {}
	for component in components:
		parameters[component._name] = component._to_json()
	return parameters

# deserializes a parameter dictionary into a list of components
def deserialize_components(parameters):
	components = []
	for component_name in parameters:
		component_parameters = parameters[component_name]
		component = deserialize_component(component_name, component_parameters)
		components.append(component)
	return components

# deserializes a json file to a component
def deserialize_component(component_name, params):
	# get types
	types = params['type'].split('.')
	parent_name, child_name = types[0], types[1]
	import_name = parent_name.lower() + 's.' + child_name.lower()
	child_module = __import__(import_name, fromlist=[child_name])
	child_type = getattr(child_module, child_name)
	# create a child object
	del params['type']
	params['name'] = component_name
	child = child_type(**params)
	return child

# keeps track of component instances
component_list = {}
def get_component(name, is_type=None):
	component = component_list[name]
	if is_type is not None:
		component.check(is_type)
	return component

# returns a list of all components created
def get_all_components():
	components = []
	for component_name in component_list:
		components.append(component_list[component_name])
	return components

# logs time and memory benchmarks
diary = {'time':{'units':'microseconds'}, 'memory':{'units':'kilobytes'}, 'event':{}}
def log_entry(master_key, key, value):
	if key not in diary[master_key]:
		diary[master_key][key] = [value]
	else:
		diary[master_key][key].append(value)

# fetches memory stored in this object, as mega bytes
def log_memory(component):
	nBytes = getsizeof(component)
	nKiloBytes = nBytes * 0.000977
	log_entry('memory', component._name, nKiloBytes)

# times all funciton calls and saves to library (microseconds)
def _timer_wrapper(method):
	def _impl(*args, **kwargs):
		module_name = method.__module__
		t1 = time()
		method_output = method(*args, **kwargs)
		if 'component' not in module_name:
			t2 = time()
			delta_t = (t2 - t1) * 1000000
			entry_name = module_name + '.' + method.__name__
			log_entry('time', entry_name, delta_t)
		return method_output
	return _impl

# sets intialization values
def _init_wrapper(init_method):

	def wrapper(self, *args, **kwargs):
		# SET ALL PUBLIC PASSED IN ARGUMENTS AS CLASS ATTRIBUTES, FETCH COMPONENTS BY NAME
		for key in kwargs:
			# convert public list of component names to private list of components
			# skip over private arguments
			if key[0] == '_' or key == 'self' or key == 'name':
				continue
			elif 'names' in key:
				component_names = kwargs[key]
				components = []
				for component_name in component_names:
					component = get_component(component_name)
					components.append(component)
				setattr(self, '_' + key.replace('_names', 's'), components)
				setattr(self, key, component_names)
			# convert public component name to private component
			elif 'name' in key:
				component_name = kwargs[key]
				component = get_component(component_name)
				setattr(self, '_' + key.replace('_name', ''), component)
				setattr(self, key, component_name)
			# set all public arguments
			else:
				setattr(self, key, kwargs[key])

		# ADD TIMER TO EACH PUBLIC CLASS METHOD
		for method in dir(self):
			if callable(getattr(self, method)) and method[0] != '_':
				setattr(self, method, _timer_wrapper(getattr(self, method)))

		# SET UNIQUE NAME AND ADD TO LIST OF COMPONENTS
		clone_id = 1
		if 'name' not in kwargs:
			name = f'{self._child().__name__}'
			try_name = f'{name}__{clone_id}' 
		else:
			name = kwargs['name']
			try_name = name
		clone_id = 2
		while try_name in component_list:
			try_name = f'{name}__{clone_id}'
			clone_id = clone_id + 1
		name = try_name
		component_list[name] = self
		self._name = name

		# CALL BASE INIT
		init_method(self, *args, **kwargs)
		print('created:', self._name)

	return wrapper

# the component class itself
class Component():
	# WRAP ALL __INIT__ WITH THIS or call super() to not set class attributes automatically
    #@_init_wrapper
	def __init__(self, name=None):

		# ADD TIMER TO EACH PUBLIC CLASS METHOD
		for method in dir(self):
			if callable(getattr(self, method)) and '__' not in method:
				setattr(self, method, _timer_wrapper(getattr(self, method)))

		# SET UNIQUE NAME AND ADD TO LIST OF COMPONENTS
		clone_id = 1
		if name is None:
			name = f'{self._child().__name__}'
			try_name = f'{name}__{clone_id}' 
		else:
			try_name = name
		clone_id = 2
		while try_name in component_list:
			try_name = f'{name}__{clone_id}'
			clone_id = clone_id + 1
		name = try_name
		component_list[name] = self
		self._name = name

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

	# runs whatever to do whatever
	def run(self):
		pass

	# kill connection, clean up as needed
	def disconnect(self):
		pass

	# resets and end of episode to prepare for next
	def reset(self, state):
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

	# turns component into a dictionary/json format - auto gets class name and self-serializable parameters
	def _to_json(self):
		params = {'type':f'{self._parent().__name__}.{self._child().__name__}'}
		variables = vars(self)
		for key in variables:
			variable = variables[key]
			if callable(variable):
				continue
			if key[0] == '_':
				continue
			if type(variable) is np.ndarray:
				variable = variable.tolist()
			params[key] = variable
		return params
      
	# checks json serialization for equality
	def __eq__(self, other):
		if self._to_json()==other._to_json():
			return True
		return False