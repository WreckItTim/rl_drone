from others.other import Other
from component import _init_wrapper
import utils
import os

# this will call save at
class Saver(Other):
	@_init_wrapper
	def __init__(self,
			environment_component,
			  modified_component,
		parent_method = 'reset',
		exectue = 'after',
		frequency = 1,
		counter = 0,
			save_components=None,
			save_variables = {},
			frequency=10,
			save_model=True,
			save_replay_buffer=True,
			save_configuration_file=True,
			save_benchmarks=True,
			write_folder=None
	):
		self.connect_priority = -999 # connects after everything else
		if write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
		if not os.path.exists(self.write_folder):
			os.makedirs(self.write_folder)

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