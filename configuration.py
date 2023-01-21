import utils
from sys import getsizeof

# saves config of components
class Configuration():
	active = None

	def __init__(self, 
			  meta, 
			  controller = None
			  add_timers=False, 
			  add_memories=False,
			  ):
		self.meta = meta
		self.add_timers = add_timers
		self.add_memories = add_memories
		self.components = {}
		self.benchmarks = {
			'time':{'units':'microseconds'}, 
			'memory':{'units':'kilobytes'},
			}
		Configuration.set_active(configuration)
		self.set_controller(controller)

	def update_meta(self, meta):
		self.meta.update(meta)

	def set_controller(self, controller):
		self.controller = controller
		controller._configuration = self

	# master resets everything for a new learning loop (fresh episode, counters, etc)
	def reset_all(self):
		for component_name in self.components:
			component = self.get_component(component_name)
			component.reset_learning()
		
	#  keep track of component benchmarks
	def log_benchmark(self, master_key, key, value):
		if key not in self.benchmarks[master_key]:
			self.benchmarks[master_key][key] = [value]
		else:
			self.benchmarks[master_key][key].append(value)

	# benchmarks all components and writes to file
	def benchmark_memory(self):
		for component_name in self.components:
			component = self.get_component(component_name)
			if component._add_memories:
				nBytes = getsizeof(component) # self.__sizeof__()
				nKiloBytes = nBytes * 0.000977
				self.benchmarks['memory'][component._name] = nKiloBytes

	# saves benchmarks
	def log_benchmarks(self, write_path=None):
		self.benchmark_memory()
		if write_path is None:
			write_path = utils.get_global_parameter('working_directory') + 'benchmarks.json'
		utils.write_json(self.benchmarks, write_path)

	# keeps track of components
	def add_component(self, component):
		self.components[component._name] = component

	# keeps track of components
	def get_component(self, component_name, is_type=None):
		if component_name not in self.components:
			utils.error(f'component named {component_name} does not exist')
			return None
		component = self.components[component_name]
		if is_type is not None:
			component.check(is_type)
		return component
		
	# connect components in order of connect_priority (positives low-to-high, default zeros, negatives high-to-low)
	def connect_all(self):
		# get priorities
		priorities = []
		component_dic = {}
		for component_name in self.components:
			component = self.get_component(component_name)
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
		start_negative_index = -1
		for index, priority in enumerate(reversed(priorities)):
			if priority < 0:
				start_negative_index = len(priorities)-index-1
				break
		for index in range(start_negative_index, -1, -1):
			for component in component_dic[priorities[index]]:
				component.connect()
		self.controller.connect()
		
	# disconnect components in order of connect_priority (positives low-to-high, default zeros, negatives high-to-low)
	def disconnect_all(self):
		# get priorities
		priorities = []
		component_dic = {}
		for component_name in self.components:
			component = self.get_component(component_name)
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
		start_negative_index = -1
		for index, priority in enumerate(reversed(priorities)):
			if priority < 0:
				start_negative_index = len(priorities)-index-1
				break
		for index in range(start_negative_index, -1, -1):
			for component in component_dic[priorities[index]]:
				component.disconnect()
		self.controller.disconnect()
		
	# serializes into a configuration json file
	def serialize(self):
		from observations.observation import Observation
		misc = {
			'nObservations':Observation.nObservations,
			}
		configuration_file = {
			'meta':self.meta,
			'misc':misc,
			'components':{}
		}
		for component_name in self.components:
			component = self.get_component(component_name)
			configuration_file['components'][component._name] = component._to_json()
		return configuration_file

	# deserializes a json file into a configuration
	@staticmethod
	def deserialize(configuration_file):
		meta = configuration_file['meta']
		misc = configuration_file['misc']
		from observations.observation import Observation
		Observation.nObservations = misc['nObservations']
		configuration = Configuration(meta)
		Configuration.set_active(configuration)
		for component_name in configuration_file['components']:
			component_arguments = configuration_file['components'][component_name]
			_ = Component.deserialize(component_name, component_arguments)
		return configuration

	def save(self, write_path=None):
		configuration_file = self.serialize()
		if write_path is None:
			write_path = utils.get_global_parameter('working_directory') + 'configuration.json'
		utils.write_json(configuration_file, write_path)

	@staticmethod
	def load(read_path):
		configuration_file = utils.read_json(read_path)
		configuration = Configuration.deserialize(configuration_file)
		return configuration
	
	@staticmethod
	def set_active(configuration):
		Configuration.active = configuration
	
	@staticmethod
	def get_active():
		return Configuration.active

from component import Component