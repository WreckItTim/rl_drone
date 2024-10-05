from maps.map import Map
import rl_utils as utils
import subprocess
import os
from component import _init_wrapper
import setup_path # need this in same directory as python code for airsim
import airsim
from others.voxels import Voxels
import psutil
import time

# handles airsim release executables
class AirSimMap(Map):

	# constructor, pass in a dictionary for settings and/or file paths to merge multiple settings .json files
	@_init_wrapper
	def __init__(self,
				 # voxels for 2d/3d numpy array represtation of objects
				 voxels_component=None,
				 # path to release (.sh/.exe) file to be launched
				 # if this is not None, will launch airsim map automatically
				 # otherwise it is up to the user to launch on their own
				 release_path:str = None,
				 # can define json-structured settings
				 settings:dict = None,
				 # or define setting files to read in from given directory
				 # will aggregate passed in json settings and all files
				 # update priority is given to the settings argument and last listed files
				 # amoung the settings must be information for which sensors to use
				 # below arg is a list of file names, see the maps/airsim_settings for examples
				 settings_directory:str = 'maps/airsim_settings/',
				 setting_files:list = ['vanilla'],
				 # optional flags to put in command line when launching
				 console_flags = None,
				 remove_animals = True,
				 ):
		super().__init__()
		if voxels_component is None:
			self._voxels = None
		self._pid = None
		self._client = None
		# get path to release executable file to launch
		if release_path is not None:
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
				utils.set_local_parameter('LocalHostIp', self.settings['LocalHostIp'])
			else:
				utils.set_local_parameter('LocalHostIp', '127.0.0.1')
			if 'ApiServerPort' in self.settings:
				utils.set_local_parameter('ApiServerPort', self.settings['ApiServerPort'])
			else:
				utils.set_local_parameter('ApiServerPort', 41451)
			# pipeline to open for console output

	def make_voxels(self,
			  # ABSOLUTE path to write to, must be absolute
			  absolute_path:str,
			  # voxel params if make new voxels (else these are set from read)
			  # user be WARNED: use a cube centered around 0,0,0 because this sh** is wonky if not
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
	def connect(self, from_crash=False):
		if from_crash:
			utils.speak('Recovering from AirSim crash...')
			self.disconnect()
		else:
			super().connect()
		
		# check OS to determine how to launch map
		OS = utils.get_local_parameter('OS')
		# set flags
		flags = ''
		if self.console_flags is not None:
			flags = ' '.join(self.console_flags)
		# launch map from OS
		if OS == 'windows':
			_release_path = self.release_path
			# send command to terminal to launch the relase executable, if can
			if os.path.exists(_release_path):
				utils.speak(f'Launching AirSim at {_release_path}')
				terminal_command = f'{_release_path} {flags} -settings=\"{self._settings_path}\"'
				utils.speak(f'Issuing command to OS: {terminal_command}')
				process = subprocess.Popen(terminal_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
				self._pid = process.pid
			else:
				utils.speak('AirSim release path DNE.')
		elif OS == 'linux':
			_release_path = self.release_path
			# send command to terminal to launch the relase executable, if can
			if os.path.exists(_release_path):
				utils.speak(f'Launching AirSim at {_release_path}')
				terminal_command = f'sh {_release_path} {flags} -settings=\"{self._settings_path}\"'
				utils.speak(f'Issuing command to OS: {terminal_command}')
				process = subprocess.Popen(terminal_command, shell=True, start_new_session=True)
				self._pid = process.pid
			else:
				utils.speak('AirSim release path DNE.')
		else:
			utils.speak('incompatible OS.')
				
		# wait for map to load
		time.sleep(10)

		# establish communication link with airsim client
		self._client = airsim.MultirotorClient(
			ip=utils.get_local_parameter('LocalHostIp'),
			port=utils.get_local_parameter('ApiServerPort'),
		)
		
		# remove animals - sad tear drop (there is a monument in the park to them, not jk)
		if self.remove_animals:
			objs = self._client.simListSceneObjects()
			animals = [name for name in objs if 'Deer' in name or 'Raccoon' in name or 'Animal' in name]
			_ = [self._client.simDestroyObject(name) for name in animals] # PETA has joined the chat
		
		self._client.confirmConnection()
		self._client.enableApiControl(True)
		self._client.armDisarm(True)
		self.reset() # this seems repetitive but needed to reset state info

	# close airsim map
	def disconnect(self):
		# this should keep child in tact to kill same process created (can handle multi in parallel)
		if self._pid is not None:
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
