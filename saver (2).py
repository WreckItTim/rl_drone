from others.other import Other
from component import _init_wrapper
import utils
import os

# this will call save at
class Saver(Other):
	@_init_wrapper
	def __init__(self,
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  order, # modify 'pre' or 'post'?
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			
			  environment_component,
			  modified_component,
		
			save_components=None,
			save_variables = {},
			frequency=10,
			save_model=True,
			save_replay_buffer=True,
			save_configuration_file=True,
			save_benchmarks=True,
			write_folder=None
	):
		super().__init__(base_component, parent_method, order, frequency, counter)

	def step(self, state):
		pass

	def reset(self, reset_state):
		pass

	def reset_learning(self, reset_state):
		pass

	def save(self):
		pass

	def load(self):
		pass

	def debug(self):
		pass

	def connect(self):
		super().connect()

	def disconnect(self):
		super().disconnect()

	def connect(self):
		super().connect()
		self._paths = []
		for component in self._save:
			component_name = component._name
			sub_folder = self.write_folder + component_name + '/'
			if not os.path.exists(sub_folder):
				os.makedirs(sub_folder)
			self._paths.append(sub_folder)
			for variable in self.save_variables[component_name]:
				component._dumps.append(variable)


	def save(self):
		for idx, component in enumerate(self._save):
			component.dump(self._paths[idx])
		if self.save_configuration_file:
			self._configuration.save(self.write_folder + 'configuration.json')
		if self.save_benchmarks:
			self._configuration.log_benchmarks(self.write_folder + 'benchmarks.json')

	def reset(self):
		if self._environment.episode_counter % self.frequency == 0:
			self.save()

	# when using the debug controller
	def debug(self):
		self.save()