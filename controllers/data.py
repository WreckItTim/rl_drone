from controllers.controller import Controller
from component import _init_wrapper
import numpy as np
import math
import rl_utils as utils
import os
import pickle
import msgpackrpc

# collects data by collecting observations
	# specify points to move to on map, along with which sensors to capture at each point
class Data(Controller):
	# constructor
	@_init_wrapper
	def __init__(self,
				 drone_component, # agent to move around map
				 sensors_components, # which sensors to capture at each point
				 map_component, # map to handle crash with
				 points, # list of points to capture data from
				 ):
		super().__init__()
		
	# runs control on components
	def run(self):
		working_directory = utils.get_local_parameter('working_directory')
		data_directory = working_directory + 'data/'
		if os.path.exists(data_directory):
			choice = utils.prompt('WARNING, folder exists and may OVERWRITE any files. enter \'y\' to continue or anything else to abort. Data path is: ' + str(data_directory))
			if choice != 'y':
				return
		else:
			os.mkdir(data_directory)
		# write points legend to file (use to reference each data point name to)
		utils.pk_write(self.points, data_directory + 'data_legend.p')
		for p_idx, point in enumerate(self.points):
			try_again = True # crash handler
			while try_again: # crash handler
				try: # crash handler
					point_name = str(p_idx)
					point_directory = data_directory + point_name + '/'
					if not os.path.exists(point_directory):
						os.mkdir(point_directory)
					self._drone.teleport(point[0], point[1], point[2], point[3], ignore_collision=True)
					for sensor in self._sensors:
						observation = sensor.step()
						path = point_directory + sensor._name
						observation.write(path)
					try_again = False # crash handler
				except msgpackrpc.error.TimeoutError as e: # crash handler
					utils.speak(str(e) + ' caught at point index # ' + str(p_idx)) # crash handler
					self._map.connect(from_crash=True) # crash handler
