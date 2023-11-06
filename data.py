import rl_utils as utils
from configuration import Configuration
import math
import numpy as np

# **** SETUP ****
out_path = 'local/runs/' # everything will be written to this parent directory
run_name = 'data1' # and to this child directory
OS = utils.setup(
	write_parent = out_path,   # will save all files to this parent folder
	run_prefix = run_name,			# and to this sub folder
	)
working_directory = utils.get_global_parameter('working_directory') # fetch working directory that this project is in 


## CONTROLLER (how to handle simulation: i.e. debug, control, reinfocement learning, SLAM, etc)
controller_type = 'Data' # options are: Train Debug Evaluate Data	
sensors = [
	'ForwardCamera',
	'ForwardDepth',
	] # fill this array with name of desired sensors to capture data from (create components below)
controller_params = {
	'drone_component' : 'Drone', # we will create a drone component to connect to our Debug controller
	'sensors_components' : sensors, # we will create a drone component to connect to our Debug controller
	'map_component' : 'Map', # we will create a drone component to connect to our Debug controller
	'points' : [], # we will create a drone component to connect to our Debug controller
}
controller = utils.get_controller( # create controller with above arguments
	controller_type = controller_type,
	controller_params = controller_params, 
)
# SET META DATA (anything you want here, this is logged to a configuration file as a json dictionary)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'repo_version': utils.get_global_parameter('repo_version'),
	'run_name': utils.get_global_parameter('run_name'),
	'timestamp': utils.get_timestamp(),
	'run_OS': utils.get_global_parameter('OS'),
	'absolute_path' : utils.get_global_parameter('absolute_path'),
	'working_directory' : working_directory,
	}


# MAKE CONFIGURATION AND COMPONENTS...
## CONFIGURATION object handles all components (will auto add below componets to this object)
configuration = Configuration(
	meta, 
	controller,
	)

# **** CREATE COMPONENTS ****
		

## MAP
from maps.airsimmap import AirSimMap
# set path to airsim release binary
if utils.get_global_parameter('OS') == 'windows':
	release_path = 'local/airsim_maps/Blocks/WindowsNoEditor/Blocks.exe'
if utils.get_global_parameter('OS') == 'linux':
	release_path = 'rl_drone/local/airsim_maps/LinuxBlocks1.9.1/LinuxNoEditor/Blocks.sh'
# add any console flags
console_flags = []
# render screen? This should be false if SSH-ing from remote (edit here or in global_parameters.json)
render_screen = utils.get_global_parameter('render_screen')
if render_screen:
	console_flags.append('-Windowed')
else:
	console_flags.append('-RenderOffscreen')
# create airsim map object
map1 = AirSimMap(
	voxels_component = 'Voxels',
	release_path = release_path,
	settings = {
		'ClockSpeed': 1, # speed up, >1, or slow down, <1, aisrim. generally dont go higher than 10 - but dependent on your hardware
		},
	setting_files = [
		'vanilla', # see maps/airsim_settings/... for different settings
		],
	console_flags = console_flags.copy(),
	name = 'Map',
)
# voxels grabs locations of objects from airsim map
# used to validate spawn and goal points (not inside an object)
# also used to visualize flight paths
from others.voxels import Voxels
distance_param = 100 # only works with cubes
Voxels(
	relative_path = working_directory + 'map_voxels.binvox',
	map_component = 'Map',
	x_length = 2 * distance_param, # total x-axis meters (split around center)
	y_length = 2 * distance_param, # total y-axis  meters (split around center)
	z_length = 2 * distance_param, # total z-axis  meters (split around center)
	name = 'Voxels',
)


#  DRONE
# create drone actor to issue commands to
# call this component from debug to takeoff
from drones.airsimdrone import AirSimDrone
AirSimDrone(
	airsim_component = 'Map',
	name='Drone',
)


# SENSORS - observation space
# forward facing scene camera
from sensors.airsimcamera import AirSimCamera
AirSimCamera(
	airsim_component = 'Map',
	image_type = 0, # scene
	is_gray = False, # scene is RGB
	as_float = False, # returns image with pixels in range: [0,1]
	name = 'ForwardCamera',
)
# forward facing depth camera
from transformers.normalize import Normalize
Normalize(
	max_input = 100, # max depth
	name = 'NormalizeDepth',
)
AirSimCamera(
	airsim_component = 'Map',
	image_type=2, # depths
	is_gray=True, # scene is RGB
	transformers_components=[
		'NormalizeDepth', # normalize depth between 0 1nad 1
		],
	name = 'ForwardDepth',
)
	
utils.speak('configuration created!')


# CONNECT COMPONENTS
configuration.connect_all()
utils.speak('all components connected.')


## POINTS (example)
# enter raw points / read from file...
# [x, y, z, yaw] (meters, meters, meters, rads:-pi,pi]) and in drone coords: up is negative z, x is forward facing from drone spawn
controller.points = [ 
	[1,2,-3,math.pi/4],
	[20,5,-6,3*math.pi/2],
	[102,24,-10,math.pi],
]
# or random gen...
# use map1.at_object_2d(x, y), map1.at_object_3d(x, y, z), and/or map1.get_roof(x, y) to find valid points on map - see maps/map.py 
controller.points = [
]
for i in range(100):
	while len(controller.points) == i:
		x = np.random.uniform(-100, 100)
		y = np.random.uniform(-100, 100)
		z = np.random.uniform(-20, 0)
		if map1.at_object_3d(x, y, z): # check if spawn in object
			continue
		yaw = np.random.uniform(-math.pi, math.pi)
		controller.points.append([x, y, z, yaw])


# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
configuration.controller.stop()