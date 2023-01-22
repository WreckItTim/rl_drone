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
			  track_vars, # which class specific variables to save [str]
			  order, # modify 'pre' or 'post'?
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = False, # will activate on first call otherwise only if % is not 0
			  write_folder = None, # will default to working_directory/component_name/
			  ):
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)
		if write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._base._name + '/'

	def connect(self):
		super().connect()
		self._base.set_save(True, self.track_vars)

	def activate(self, state=None):
		if self.check_counter():
			_write_folder = self.write_folder + self._base.write_prefix()
			if not os.path.exists(_write_folder):
				os.makedirs(_write_folder)
			self._base.save(_write_folder, state)