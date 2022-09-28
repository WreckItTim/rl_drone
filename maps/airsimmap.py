from maps.map import Map
import utils
import subprocess
import os
from component import _init_wrapper
import setup_path # need this in same directory as python code for airsim
import airsim
from datastructs.voxels import Voxels

# handles airsim release executables
class AirSimMap(Map):

	# constructor, pass in a dictionary for settings and/or file paths to merge multiple settings .json files
	@_init_wrapper
	def __init__(self,
				 # voxels for 2d/3d numpy array represtation of objects
				 voxels_component=None,
				 ):
		super().__init__()

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
		# prompt user to confirm when launch is successful (can launch manually if needs be)
		print(f'Manually launch AirSim map, and press any key when complete...')
		x = input()

	# close airsim map
	def disconnect(self):
		# send command to terminal to kill the relase executable
		terminal_command = 'taskkill /f /im ' + self.release_name
		subprocess.call(terminal_command)
	
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
