from modifiers.modifier import Modifier
from component import _init_wrapper
import rl_utils as utils
import os
import wandb

# this will call save at
class Saver(Modifier):
	@_init_wrapper
	def __init__(self,
			base_component, # componet with method to modify
			parent_method, # name of parent method to modify
			track_vars, # which class specific variables to save [str]
			order, # modify 'pre' or 'post'?
			include_counter = False, # includes activation counter in write file name
				# can use include_counter=True to overwrite model each epoch
			save_config = False, # saves config file with every activate
			save_benchmarks = False, # saves timer/mem benchmarks with every activate
			write_folder = None, # will default to working_directory/component_name/
			on_evaluate = True, # toggle to run modifier on evaluation environ
			on_train = True, # toggle to run modifier on train environ
			frequency = 1, # use modifiation after how many calls to parent method?
			counter = 0, # keepts track of number of calls to parent method
			activation_counter = 0, # keeps track of number of times activated
			activate_on_first = True, # will activate on first call otherwise only if % is not 0
			save_on_exit = True, # save all when program ends
			):
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)

	def connect(self, state=None):
		super().connect(state)
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._base._name + '/'
		self._base.set_save(True, self.track_vars)

	def disconnect(self, state=None):
		super().disconnect(state)
		if state is None:
			state = {}
		state['activate_saver'] = True
		if self.save_on_exit:
			self.activate(state)

	def activate(self, state=None):
		if state is None:
			state = {}
		if 'activate_saver' not in state:
			state['activate_saver'] = False
		if state['activate_saver'] or self.check_counter(state):
			_write_folder = self.write_folder + self._base.write_prefix()
			if not os.path.exists(_write_folder):
				os.makedirs(_write_folder)
			if state is None: 
				state = {}
			state['write_folder'] = _write_folder
			self._base.save(state)
			if self.save_config:
				self._configuration.save()
				if wandb.run is not None:
					wandb.save(utils.get_global_parameter('working_directory') + 'configuration.json')
			if self.save_benchmarks:
				self._configuration.save_benchmarks()
			self.activation_counter += 1
