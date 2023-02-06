from modifiers.modifier import Modifier
from component import _init_wrapper
import utils
import os
import nvidia_smi
import psutil

# this will track resource usage 
class Tracker(Modifier):
	@_init_wrapper
	def __init__(self,
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  track_vars, # which variables to track, from list:
			  # gpu cpu ram
			  order, # modify 'pre' or 'post'?
			  write_path, # path to write log to
			  on_evaluate = True, # toggle to run modifier on evaluation environ
			  on_train = True, # toggle to run modifier on train environ
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = True, # will activate on first call otherwise only if % is not 0
			  ):
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)

	def connect(self, state=None):
		super().connect(state)
		self._log = {tv:{} for tv in self.track_vars}
		if 'gpu' in self._log:
			nvidia_smi.nvmlInit()
			self._nGPUs = nvidia_smi.nvmlDeviceGetCount()
			for i in range(self._nGPUs):
				self._log['gpu'][i] = {}

	def disconnect(self, state=None):
		super().disconnec(state)
		if 'gpu' in self._log:
			nvidia_smi.nvmlShutdown()

	def activate(self, state=None):
		if self.check_counter(state):
			timestamp = utils.get_timestamp()
			if 'gpu' in self._log:
				for i in range(self._nGPUs):
					handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
					util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
					#mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
					self._log['gpu'][i][timestamp] = {
						'util':util.gpu,
						'mem':util.memory,
					}
			if 'ram' in self._log:
				ram = psutil.virtual_memory()[2]
				self._log['ram'][timestamp] = ram
			if 'cpu' in self._log:
				cpu = psutil.cpu_percent()
				self._log['cpu'][timestamp] = cpu
			utils.write_json(self._log, self.write_path)
