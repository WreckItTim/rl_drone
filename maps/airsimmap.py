from maps.map import Map
import utils
import subprocess
import os
from component import _init_wrapper
import setup_path # need this in same directory as python code for airsim
import airsim
from datastructs.voxels import Voxels
import time
import signal
import psutil

# handles airsim release executables
class AirSimMap(Map):

	# constructor, pass in a dictionary for settings and/or file paths to merge multiple settings .json files
	@_init_wrapper
	def __init__(self,
				 # voxels for 2d/3d numpy array represtation of objects
				 voxels_component=None,
				 # can define json-structured settings
				 settings:dict = None,
				 # or define setting files to read in from given directory
				 settings_directory:str = 'maps/airsim_settings/',
				 # will aggregate passed in json settings and all files
				 # update priority is given to the settings argument and last listed files
				 # amoung the settings must be information for which sensors to use
				 # below arg is a list of file names, see the maps/airsim_settings for examples
				 setting_files:list = ['vanilla'],
				 # directory to release .exe file to be launched
				 release_directory:str = None,
				 # name of release .exe/.sh file to be launched, inside relase_directory
				 release_name:str = None,
				 # optional flags to put in command line when launching
				 console_flags = None,
				 # controls if to make a voxels object on connect
				 make_voxels_on_connect = False,
				 ):
		super().__init__()
		# get path to release executable file to launch
		self._release_path = os.path.join(release_directory, release_name, release_name)
		# create setting dictionary
		self.settings = {}
		if settings is not None:
			self.settings = settings.copy()
		# read in any other settings files
		other_settings = {}
		if setting_files is not None:
			other_settings = self.read_settings(settings_directory, setting_files)
		# merge all settings
		self.settings.update(other_settings)
		# write to temp file to be read in when launching realease executable
		self._settings_path = os.getcwd() + '/temp/overwrite_settings.json'
		self.write_settings(self.settings, self._settings_path)
		if 'LocalHostIp' in self.settings:
			utils.set_global_parameter('LocalHostIp', self.settings['LocalHostIp'])
		else:
			utils.set_global_parameter('LocalHostIp', '127.0.0.1')
		# pipeline to open for console output
		self._pid = None

	def make_voxels(self,
			  # ABSOLUTE path to right to, must be absolute
			  absolute_path:str,
			  # voxel params if make new voxels (else these are set from read)
			  center = [0,0,0], # in meters
			  resolution = 1, # in meters
			  x_length = 200, # total x-axis meters (split around center)
			  y_length = 200, # total y-axis  meters (split around center)
			  z_length = 200, # total z-axis  meters (split around center)
	):
		client = airsim.VehicleClient()
		center = airsim.Vector3r(center[0], center[1], center[2])
		# must create voxel using an absolute path
		client.simCreateVoxelGrid(center, 
									x_length, 
									y_length, 
									z_length, 
									resolution, 
									absolute_path,
									)

	# launch airsim map
	def connect(self):
		super().connect()
		OS = utils.get_global_parameter('OS')
		flags = ''
		if self.console_flags is not None:
			flags = ' '.join(self.console_flags)
		if OS == 'Windows':
			# send command to terminal to launch the relase executable, if can
			if os.path.exists(self._release_path):
				print(f'Launching AirSim at {self._release_path}')
				terminal_command = f'{self._release_path}.exe {flags} -settings=\"{self._settings_path}\"'
				process = subprocess.Popen(terminal_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
				self._pid = process.pid
			else:
				print('Please manually launch Airsim.')
		if OS == 'Linux':
			# send command to terminal to launch the relase executable, if can
			if os.path.exists(self._release_path):
				print(f'Launching AirSim at {self._release_path}')
				terminal_command = f'sh {self._release_path}.sh {flags} -settings=\"{self._settings_path}\"'
				process = subprocess.Popen(terminal_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
				self._pid = process.pid
			else:
				print('Please manually launch Airsim.')
		if OS == 'Darwin':
			print('Please manually launch Airsim.')
		# prompt user to confirm when launch is successful (can launch manually if needs be)
		print(f'Send any key when AirSim is fully launched, this make take several minutes....')
		x = input()

	# close airsim map
	def disconnect(self):
		print('DISCONNECT AIRSIMMAP')
		# this should keep child in tact to kill same process created (can handle multi in parallel)
		if self._pid is not None:
			print('attempting to kill pid', self._pid)
			x = input()
			parent = psutil.Process(self._pid)
			for child in parent.children(recursive=True):
				child.kill()
			parent.kill()
	
	# read several json files with Airsim settings and merge
	def read_settings(self, settings_directory, setting_files):
		merged_settings = {}
		for setting_file in setting_files:
			setting_path = os.path.join(settings_directory, setting_file) + '.json'
			setting = utils.read_json(setting_path)
			merged_settings.update(setting)
		return merged_settings

	# write a json settings dictionary to file
	def write_settings(self, merged_settings, settings_path):
		utils.write_json(merged_settings, settings_path)