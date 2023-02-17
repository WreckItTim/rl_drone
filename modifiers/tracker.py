from modifiers.modifier import Modifier
from component import _init_wrapper
import utils
import os
import nvidia_smi
import psutil
from nvitop import Device, GpuProcess

# this will track resource usage 
class Tracker(Modifier):
	@_init_wrapper
	def __init__(self,
			  base_component, # componet with method to modify
			  parent_method, # name of parent method to modify
			  track_vars, # which variables to track, from list:
			  # gpu cpu ram proc
			  order, # modify 'pre' or 'post'?
			  save_every, # every number of activations
			  write_folder = None, # path to write log to
			  nActivations = 0,
			  nParts = 0,
			  on_evaluate = True, # toggle to run modifier on evaluation environ
			  on_train = True, # toggle to run modifier on train environ
			  frequency = 1, # use modifiation after how many calls to parent method?
			  counter = 0, # keepts track of number of calls to parent method
			  activate_on_first = True, # will activate on first call otherwise only if % is not 0
			  ):
		super().__init__(base_component, parent_method, order, frequency, counter, activate_on_first)
		self._CONVERSION = float(1024**3) # bytes to GB

	def reset_log(self):
		self._log = {}
		for tv in self.track_vars:
			if tv in ['gpu']:
				self._log['sys_gpu%'] = {}
				nvidia_smi.nvmlInit()
				self._nGPUs = nvidia_smi.nvmlDeviceGetCount()
				for i in range(self._nGPUs):
					self._log['sys_gpu%'][i] = {}
			elif tv in ['cpu', 'ram']:
				self._log['sys_' + tv + '%'] = {}
			elif tv in ['proc']:
				self._log[tv] = {}

	def connect(self, state=None):
		super().connect(state)
		self._devices = Device.all()
		if self.write_folder is None:
			self.write_folder = utils.get_global_parameter('working_directory')
			self.write_folder += self._name + '/'
		if not os.path.exists(self.write_folder):
				os.makedirs(self.write_folder)
		self.reset_log()

	def disconnect(self, state=None):
		super().disconnect(state)
		if 'gpu' in self._log:
			nvidia_smi.nvmlShutdown()

	def activate(self, state=None):
		if self.check_counter(state):
			timestamp = utils.get_timestamp()
			if 'gpu' in self.track_vars:
				for i in range(self._nGPUs):
					handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
					util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
					#mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
					self._log['sys_gpu%'][i][timestamp] = {
						'util':util.gpu,
						'mem':util.memory,
					}
			if 'ram' in self.track_vars:
				ram = psutil.virtual_memory()[2]
				self._log['sys_ram%'][timestamp] = ram
			if 'cpu' in self.track_vars:
				cpu = psutil.cpu_percent()
				self._log['sys_cpu%'][timestamp] = cpu
			if 'proc' in self.track_vars:
				proc_info = []
				# cpu processes
				for proc in psutil.process_iter():
					try:
						this_info = proc.as_dict(attrs=[
							'name', 
							'username', 
							'cpu_percent',
							])
						this_info['ram'] = proc.memory_info().rss / self._CONVERSION
						if this_info['ram'] >= 1 or this_info['cpu_percent'] >= 100:
							this_info['ram'] = str(round(this_info['ram'],2)) + 'GB'
							proc_info.append(this_info)
					except:
						pass
				# gpu processes
				for device in self._devices:
					processes = device.processes()
					snapshots = GpuProcess.take_snapshots(processes.values(), failsafe=True)
					for snapshot in snapshots:
						this_info = {
							'name' : snapshot.name,
							'username' : snapshot.username,
							'host_memory' : snapshot.host_memory_human,
							'memory_percent' : snapshot.memory_percent,
							'cpu_percent' : snapshot.cpu_percent,
							'gpu_memory' : snapshot.gpu_memory_human,
							'gpu_memory_percent' : snapshot.gpu_memory_percent,
							'gpu_sm_utilization' : snapshot.gpu_sm_utilization,
							'running_time_in_seconds' : snapshot.running_time_in_seconds,
						}
						proc_info.append(this_info)

				self._log['proc'][timestamp] = proc_info.copy()
			write_path = self.write_folder + 'temp.json'
			utils.write_json(self._log, write_path)
			self.nActivations += 1
			if self.nActivations % self.save_every == 0:
				self.nParts += 1
				write_path = self.write_folder + 'Part' + str(self.nParts) + '.json'
				utils.write_json(self._log, write_path)
				self.reset_log()
