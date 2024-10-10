import rl_utils as utils
from configuration import Configuration
import os
import pickle
import numpy as np

# test params
run_name = 'example_dqn' #  'example_td3'
airsim_release_path = 'local/airsim_maps/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
render_screen = True # render graphics to screen?
clock_speed = 8 # adjust this to your device (8 is stable on most modern devices)
device = 'cuda:0' # run model on this device
read_config_file = 'local/runs/'+run_name+'/configuration.json' # make components from this config file
read_model_file = 'local/runs/'+run_name+'/model.zip' # sb3 model that contains actor to use
write_test_dir = 'local/runs/'+run_name+'/test/' # write test results to this directory
astar_paths_file = 'astar_paths/Blocks_2d_test.p' # read these Astar shortest paths to evaluate config on
num_evals_per_sublevel = 1 # how many evaluations to do from above path, there can be up to  hundreds of sublevels depending on the levels file
start_level, end_level = 1, 10 # range of levels to test on from astar paths

# setup
working_directory = write_test_dir
utils.setup(working_directory)
astar_paths = pickle.load(open(astar_paths_file, 'rb'))
num_sublevels = np.sum([len(astar_paths['levels'][level]) for level in range(start_level,end_level+1)])
num_episodes = 4# int(num_evals_per_sublevel * num_sublevels)
console_flags = ['-Windowed'] # launch in windowed mode if rendering
if not render_screen:
	console_flags = ['-RenderOffscreen'] # else do not render

# make config ...

## CONTROLLER
from controllers.test import Test
controller = Test(
		environment_component = 'Environment', # environment to run test in
		model_component = 'Model', # used to make predictions
		results_directory = working_directory,
		num_episodes = num_episodes,
	)
# SET META DATA (anything you want here, just writes to config file as a dict)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'timestamp': utils.get_timestamp(),
	'run_OS': utils.get_local_parameter('OS'),
	'absolute_path' : utils.get_local_parameter('absolute_path'),
	'working_directory' : working_directory,
	}

## read old CONFIGURATION 
configuration = Configuration.load(
	read_config_file, # read all components in this config file
	controller, # evaluate
	read_modifiers=False, # no modifiers from train data - we will make new ones
	skip_components = ['Map', 'Spawner'], # change map we launch in, change how we spawn
	change_params={'device':device} # change device we run on
	)
configuration.update_meta(meta)

## MAP
from maps.airsimmap import AirSimMap
# create airsim map object
AirSimMap(
	release_path = airsim_release_path, 
	settings = {
		'ClockSpeed': clock_speed,
		},
	setting_files = [
		'lightweight', # see maps/airsim_settings
		],
	console_flags = console_flags,
	name = 'Map',
)
# SPAWNER - spawn drone at first point in each astar path with target of last point
from spawners.levels import Levels
Levels(
	drone_component = 'Drone',
	levels_path = astar_paths_file,
	paths_per_sublevel = num_evals_per_sublevel,
	start_level = start_level,
	level = start_level,
	max_level = end_level,
	name = 'Spawner',
)
# SAVERS - save observations and states at each step
from modifiers.saver import Saver
# save Train states and observations
Saver(
	base_component = 'Environment',
	parent_method = 'end',
	track_vars = [
				'observations', 
				'states',
				],
	write_folder = working_directory + 'states/',
	order = 'post',
	save_config = False,
	save_benchmarks = False,
	frequency = num_episodes,
	name='Saver',
)
# MODEL - change read path
model_component = configuration.get_component('Model')
model_component.read_model_path = read_model_file

# CONNECT new and old COMPONENTS
configuration.connect_all()

# WRITE combined new+old CONFIGURATION
configuration.save(write_path=utils.get_local_parameter('working_directory') + 'configuration.json')

# RUN CONTROLLER
configuration.controller.run()

# DISCONNECT COMPONENTS
configuration.disconnect_all()
