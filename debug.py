import rl_utils as utils
from configuration import Configuration
import math

# **** SETUP ****
out_path = 'local/runs/' # everything will be written to this parent directory
run_name = 'debug1' # and to this child directory
OS = utils.setup(
	write_parent = out_path,   # will save all files to this parent folder
	run_prefix = run_name,			# and to this sub folder
	)
working_directory = utils.get_global_parameter('working_directory') # fetch working directory that this project is in 


## CONTROLLER (how to handle simulation: i.e. debug, control, reinfocement learning, SLAM, etc)
controller_type = 'Debug' # some options are: Train Debug Drift Evaluate Data	
controller_params = {
	'drone_component' : 'Drone', # we will create a drone component to connect to our Debug controller
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

# either read saved config file to read in all components or create new configuration to make new components manually

# either READ CONFIGURATION from saved config file
read_configuration = False # set this to true to read in an old configuration from file
read_configuration_path = '.../configuration.json' # set path to config file
if read_configuration:
	# load configuration file and create object to save and connect components
	configuration = Configuration.load(read_configuration_path, controller)
	configuration.update_meta(meta)
		

# or MAKE NEW CONFIGURATION to manually make components
else:
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
	AirSimMap(
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


	#  DRONE
	# create drone actor to issue commands to
	# call this component from debug to takeoff
	from drones.airsimdrone import AirSimDrone
	AirSimDrone(
		airsim_component = 'Map',
		name='Drone',
	)


	## ACTIONS - action space		
	# actions are scaled by the "base" value between -1 and 1
	actions = []
	# this will move foward of backward
	from actions.move import Move 
	actions.append(Move(
		drone_component = 'Drone', 
		base_x_rel = 10, # move up to 10 meters forward 
		adjust_for_yaw = True, # makes movements relative to yaw
		name = 'ForwardAction',
	))
	# this will rotat yaw of dronw
	from actions.rotate import Rotate 
	actions.append(Rotate(
		drone_component = 'Drone',  
		base_yaw = math.pi, # move up to 180 degrees
		name = 'RotateAction',
	))
	# for vertical motion: negative is up and positive is down (drone coordinates)
	from actions.move import Move 
	actions.append(Move(
		drone_component = 'Drone', 
		base_z_rel = 10,  # move up to 10 meters vertical (positive = upward, negative = downward)
		adjust_for_yaw = True, # makes movements relative to yaw
		name = 'VerticalAction',
	))
	# ACTOR 
	# teleporter will directly move to position (stable, quicker)
	from actors.teleporter import Teleporter
	Teleporter(
		drone_component = 'Drone',
		actions_components = actions,
		name='NaviActor',
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

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
configuration.controller.stop()