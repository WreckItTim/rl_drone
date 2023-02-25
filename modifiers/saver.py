from modifiers.modifier import Modifier
from component import _init_wrapper
import utils
import os

# this will call save at
class Saver(Modifier):
	@_init_wrapper
	def __init__(self,
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  track_vars, # which class specific variables to save [str]
			  order, # modify 'pre' or 'post'?
			  include_counter = False, # includes counter in write file name
			  	# can use include_counter=True to overwrite model each epoch
			  save_config = False, # saves config file with every activate
			  save_benchmarks = False, # saves timer/mem benchmarks with every activate
			  write_folder = None, # will default to working_directory/component_name/
			  on_evaluate = True, # toggle to run modifier on evaluation environ
			  on_train = True, # toggle to run modifier on train environ
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = True, # will activate on first call otherwise only if % is not 0
			  ):
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)

	def connect(self, state=None):
		super().connect(state)
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._base._name + '/'
		self._base.set_save(True, self.track_vars)

	def activate(self, state=None):
		if self.check_counter(state):
			_write_folder = self.write_folder + self._base.write_prefix()
			if self.include_counter:
				_write_folder += 'counter' + str(self.counter) + '_'
			if not os.path.exists(_write_folder):
				os.makedirs(_write_folder)
			if state is None: 
				state = {}
			state['write_folder'] = _write_folder
			self._base.save(state)
			if self.save_config:
				self._configuration.save()
			if self.save_benchmarks:
				self._configuration.save_benchmarks()
