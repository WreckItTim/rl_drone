import rl_utils as utils
from configuration import Configuration
import math
import numpy as np


##  **** SETUP ****

# set path to precompiled airsim release binary
release_path = 'local/airsim_maps/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'

# render screen? This should be false if SSH-ing from remote (edit here or in global_parameters.json)
render_screen = True

# set path to read rooftops from to determine collidable surface positions
rooftops_path = 'rooftops/Blocks.p' # match to map or use voxels if not available

# write run files to this directory
working_directory = 'local/runs/example_data/'

# setup for run, set system vars and prepare file system
utils.setup(working_directory, overwrite_directory=True) # WARNING: overwrite_directory will clear all old data in this folder

# make data controller to teleport to each specified point and capture all data from given sensors
sensors = [ # we will create these sensor components below (pass in string names here, this is to handle connection priority later)
	'ForwardCamera',
	'ForwardDepth',
	] # fill this array with name of desired sensors to capture data from (create components below)
from controllers.data import Data
controller = Data(
	drone_component = 'Drone', # drone to issue commands to move around map
	sensors_components = sensors, # list of sensors to fetch data
	map_component = 'Map', # map to move around in
	points = [], # list of points to visit
)
# SET META DATA (anything you want here, just writes to config file as a dict)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'timestamp': utils.get_timestamp(),
	'run_OS': utils.get_local_parameter('OS'),
	'absolute_path' : utils.get_local_parameter('absolute_path'),
	'working_directory' : working_directory,
	}

# make configuration object to add components to
configuration = Configuration(meta, controller)


##  **** CREATE COMPONENTS ****

## create airsim map to query commands to
from maps.airsimmap import AirSimMap
# add any console flags
console_flags = []
if render_screen:
	console_flags.append('-Windowed')
else:
	console_flags.append('-RenderOffscreen')
# create airsim map object
map1 = AirSimMap(
	release_path = release_path,
	settings = {
		'ClockSpeed': 1, # speed up, >1, or slow down, <1. For aisrim, generally don't go higher than 10 - but this is dependent on your setup
		},
	setting_files = [
		'vanilla', # see maps/airsim_settings/... for different settings
		],
	console_flags = console_flags.copy(),
	name = 'Map',
)

# read rooftops data struct to determine z-height of nearest collidable object below or above x,y coords
from others.rooftops import Rooftops
rooftops = Rooftops(
	read_path = rooftops_path,
	name = 'Rooftops',
)

# create drone actor to move around map
from drones.airsimdrone import AirSimDrone
AirSimDrone(
	airsim_component = 'Map',
	name='Drone',
)

## create sensors to collect data from at each point
# forward facing scene camera that returns an array with size [3, width, height] and dtype uint8. 
	# width and height are set in the airsim settings json file
from sensors.airsimcamera import AirSimCamera
AirSimCamera(
	airsim_component = 'Map',
	image_type = 0,
	as_float = False, # forces uint8
	name = 'ForwardCamera',
)
# forward facing depth camera
# depth camera returns an array with nearest distance to an object for each pixel as a float in the range [0, inf) with size [width, height]
AirSimCamera(
	airsim_component = 'Map',
	image_type = 2,
	name = 'ForwardDepth',
)

# all components created
utils.speak('configuration created!')

# connect all components in priority order
configuration.connect_all()
utils.speak('all components connected.')


## **** SPECIFY POINTS TO COLLECT DATA FROM ****

## specify which points to grab data from on map
# here are some dummy values as an example...
# [x, y, z, yaw] (meters, meters, meters, rads:-pi,pi]) and in drone coords: 
	# x is positive forward facing from drone spawned at origin with 0 yaw
	# y is positive to the right from drone spawned at origin with 0 yaw
	# the upwards direction is negative z, not positive
	# yaw is radians rotated clockwise along the x,y plane
dummy_points = [ 
	[1,2,-3,math.pi/4],
	[20,5,-6,3*math.pi/2],
	[102,24,-10,math.pi],
]
# here is how to generate random poitns and check if an object is at that point...
# use rooftops object to detect z-coordinate of collidable surface at x,y location
random_points = []
n_random_points = 100
for i in range(n_random_points):
	# randomly create point until not inside of an object
	while len(random_points) == i:
		x = np.random.uniform(-125, 125)
		y = np.random.uniform(-125, 125)
		z = np.random.uniform(-20, 0) # max height of 20 meters
		in_object = rooftops.in_object(x, y, z)
		if in_object: # check if point in object
			continue
		yaw = np.random.uniform(-math.pi, math.pi) # random full motion of yaw
		random_points.append([x, y, z, yaw])
controller.points = dummy_points # random_points


# write configuration to file (to be viewed or loaded later)
configuration.save()

# run controller to collect all data
configuration.controller.run()

# clean up shop when finished
configuration.controller.stop()
