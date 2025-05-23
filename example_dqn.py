import rl_utils as utils
from configuration import Configuration
import math
import os

# path to airsim release file - precompiled binary
airsim_release_path = 'local/airsim_maps/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'

# render_screen=True will render AirSim graphics to screen using Unreal engine, 
	# set to false if running from SSH or want to save resources
render_screen = True

# clock speed to run simulation at 
clock_speed = 8

# set path to read rooftops from to determine collidable surface positions
rooftops_path = 'rooftops/Blocks.p' # match to map or use voxels if not available

# write run files to this directory
working_directory = 'local/runs/example_dqn/'

# load pytorch models onto this device
device = 'cuda:0'

# set policy type (either mlp for vec inputs, CNN for img inputs, or multi for both)
policy_type = 'MlpPolicy' # MlpPolicy CnnPolicy MultiInputPolicy

# how many previous timesteps to capture at each step
nTimesteps = 1

# may need to change according to your available memory
buffer_size = 10_000

# WARNING: overwrite_directory will clear all old data in this folder
overwrite_directory = False

# check if we continue training from previous run
continue_training = False # set to True will load previous model and config.json file

# setup for run, set system vars and prepare file system
utils.setup(
	working_directory, 
	overwrite_directory=overwrite_directory,
	)

# make controller to run configuration on (we will train a model)
from controllers.train import Train
controller = Train(
	model_component = 'Model',
	environment_component = 'Environment',
	continue_training = continue_training,
	)

# continue training will load runs folder and pick up where it left off
if continue_training:
	# load configuration file and create object to save and connect components
	read_configuration_path = working_directory + 'configuration.json'
	configuration = Configuration.load(read_configuration_path, controller)
	meta = configuration.meta
	if 'continued_from' in   configuration.meta:      
		meta['continued_from'] = configuration.meta['continued_from'].copy() + [configuration.meta['timestamp']]
	else:            
		meta['continued_from'] = [configuration.meta['timestamp']]
		configuration.update_meta(meta)
	utils.speak('continuing training...')

# if not continuing training then make a brand spaking new config
else:
	# set meta data (anything you want here, just writes to config file as a dict)
	meta = {
		'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
		'timestamp': utils.get_timestamp(),
		'run_OS': utils.get_local_parameter('OS'),
		'absolute_path' : utils.get_local_parameter('absolute_path'),
		'working_directory' : working_directory,
		}

	## make a new configuration file to add components to 
		# this obj will be used to serialize components, and log experiment history
		# any components created after making this configuration file will auto be added to it
		# components use the name of other components which will be created and connected later
		# this is done to handle different objects which need to be connected to eachother and connected in different orders
		# there is a baked in priority que for connecting components if you need to handle this differently
	configuration = Configuration(
		meta, 
		controller,
		)

	## create environment component to handle step() and reset() for DRL model training
	from environments.goalenv import GoalEnv
	GoalEnv(
		drone_component = 'Drone', 
		actor_component = 'Actor', 
		observer_component = 'Observer', 
		rewarder_component = 'Rewarder',
		model_component = 'Model',
		map_component = 'Map',
		spawner_component = 'Spawner',
		name = 'Environment',
		)

	## create map component to handle things such as simulation phsyics and graphics rendering
		# we will use Airsim here
	from maps.airsimmap import AirSimMap
	console_flags = [] # add any desired airsim commands here
	if render_screen:
		console_flags.append('-Windowed') # render in windowed mode for more flexibility
	else:
		console_flags.append('-RenderOffscreen') # do not render graphics, only handle logic in background
	# create airsim map object
	AirSimMap(
		release_path = airsim_release_path,
		settings = {
			'ClockSpeed': clock_speed, # reduce this value if experiencing lag
			},
		setting_files = [
			'lightweight', # see maps/airsim_settings
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
	# component that sets bounds drone can spawn in for training
	# this depends on two things:
		# (1) the dimensions of the map being used
		# (2) the target area being used for training
	from others.boundscube import BoundsCube
	BoundsCube(
		center = [0, 0, 0],
		x = [0, 100], # training bounds are in top half of map, bottom half is reserved for testing
		y = [-140, 140], # full horizontal range is used for training
		z = [-30, -2], # tallest object in blocks in 23 meters, do not spawn less than 2 meters off ground
		name = 'MapBounds'
		)	

	# drone controller component - we will use AirSim
		# this can also be real world drone controller like Tello
	from drones.airsimdrone import AirSimDrone
	AirSimDrone(
		airsim_component = 'Map',
		name = 'Drone',
		)

	## REWARD FUNCTION
	# we will reward moving closer, reward reaching goal, penalize too many steps, and penalize collisions
	rewards = []
	# heavy penalty for collision
	from rewards.collision import Collision
	Collision(
		drone_component = 'Drone',
		name = 'CollisionReward',
		)
	rewards.append('CollisionReward')
	# heavy reward for reaching goal
	from rewards.goal import Goal
	Goal(
		drone_component = 'Drone',
		goal_component = 'Spawner',
		include_z = False, # includes z in distance calculations
		tolerance = 2, # must reach goal within this many meters
		terminate = True, # we are terminating this example when the drone realizes it reached the goal, collides, or reaches max
		name = 'GoalReward',
		)
	rewards.append('GoalReward')
	# heavy penalty for using too many steps
	from rewards.maxsteps import MaxSteps
	MaxSteps(
		name = 'MaxStepsReward',
		update_steps = True, # update the maximum number of steps based on distance to goal
		max_steps = 4, # base number of steps, will scale with further goal
		max_max = 50, # absolute max number of steps, regardless of scaling from further goals
		)
	rewards.append('MaxStepsReward')
	# intermediate penalty for using more steps
	from rewards.steps import Steps
	Steps(
		name = 'StepsReward',
		)
	rewards.append('StepsReward')
	# intermediate reward for approaching goal
	from rewards.distance import Distance
	Distance(
		drone_component = 'Drone',
		goal_component = 'Spawner',
		include_z = True, # includes z in distance calculations
		name = 'DistanceReward',
		)
	rewards.append('DistanceReward')
	# REWARDER
	from rewarders.schema import Schema
	Schema(
		rewards_components = rewards,
		reward_weights = [1, 10, 1, .1, .1], 
		name = 'Rewarder',
		)

	## ACTION SPACE
	# we will just move forward and rotate for this example
	actions = []
	from actions.fixedmove import FixedMove 
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 1, # can move forward up to 10 meters
		adjust_for_yaw = True, # this adjusts movement based on current yaw
		name = 'MoveForward1',
		)
	actions.append('MoveForward1')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 2, # can move forward up to 10 meters
		adjust_for_yaw = True, # this adjusts movement based on current yaw
		name = 'MoveForward2',
		)
	actions.append('MoveForward2')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 4, # can move forward up to 10 meters
		adjust_for_yaw = True, # this adjusts movement based on current yaw
		name = 'MoveForward4',
		)
	actions.append('MoveForward4')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 8, # can move forward up to 10 meters
		adjust_for_yaw = True, # this adjusts movement based on current yaw
		name = 'MoveForward8',
		)
	actions.append('MoveForward8')
	from actions.fixedrotate import FixedRotate 
	FixedRotate(
		drone_component = 'Drone',  
		yaw_diff = math.pi/2, # can rotate at 90 deg increments
		name = 'RotateRight',
		)
	actions.append('RotateRight')
	FixedRotate(
		drone_component = 'Drone',  
		yaw_diff = -1*math.pi/2, # can rotate at 90 deg increments
		name = 'RotateLeft',
		)
	actions.append('RotateLeft')
	## ACTOR
	from actors.teleporterdiscrete import TeleporterDiscrete
	# we use a teleporter here because it is quicker and more stable
		# it will check collisions between current point and telported point then move directly to that location
	TeleporterDiscrete(
		drone_component = 'Drone',
		actions_components = actions,
		name = 'Actor',
		)

	## OBSERVATION SPACE
	# we will use the relative displacement between drone and goal, and front-facing depth maps
	# we will use the T0many most recent observations concatenated toghter, for this example T = 4
	# TRANSFORMERS
	from transformers.normalize import Normalize
	Normalize(
		min_input = -1*math.pi, # min angle
		max_input = math.pi, # max angle
		name = 'NormalizeOrientation',
		)
	Normalize(
		min_input = 1, # in front of sensor
		max_input = 100, # horizon
		left = 0,
		name = 'NormalizeDistance',
		)
	Normalize(
		min_input = 1, # in front of sensor
		max_input = 100, # horizon
		min_output = 1,
		max_output = 255,
		left = 0,
		name = 'NormalizeDistanceImg',
		)
	# SENSORS
	# sense horz distance to goal
	from sensors.distance import Distance
	Distance(
		misc_component = 'Drone',
		misc2_component = 'Spawner',
		include_z = False,
		prefix = 'drone_to_goal',
		transformers_components = [
			'NormalizeDistance',
			], 
		name = 'GoalDistance',
		)
	# sense yaw difference to goal 
	from sensors.orientation import Orientation
	Orientation(
		misc_component = 'Drone',
		misc2_component = 'Spawner',
		prefix = 'drone_to_goal',
		transformers_components = [
			'NormalizeOrientation',
			],
		name = 'GoalOrientation',
		)
	depth_transformers_components = []
	if policy_type in ['MlpPolicy']:
		# get flattened depth map (obsfucated front facing distance sensors)
		from transformers.resizeflat import ResizeFlat
		# airsim camera's image pixel size default is  "Width": 256, "Height": 144,
		max_cols = [32*(i+1) for i in range(8)] # splits depth map by columns
		max_rows = [24*(i+1) for i in range(6)] # splits depth map by rows
		ResizeFlat(
			max_cols = max_cols,
			max_rows = max_rows,
			name = 'ResizeFlat',
			)
		depth_transformers_components.append('ResizeFlat')
		depth_transformers_components.append('NormalizeDistance')
	else:
		depth_transformers_components.append('NormalizeDistanceImg')
	from sensors.airsimcamera import AirSimCamera
	AirSimCamera(
		airsim_component = 'Map',
		transformers_components = depth_transformers_components,
		name = 'DepthMap',
		)
	# OBSERVER
	# currently must count vector size of sensor output
	vector_sensors = []
	vector_length = 0
	if policy_type in ['MlpPolicy']:
		vector_sensors.append('DepthMap')
		vector_length +=  len(max_cols) * len(max_rows) # several more vector elements
	vector_sensors.append('GoalDistance')
	vector_length +=  1
	vector_sensors.append('GoalOrientation')
	vector_length +=  1
	from observers.single import Single
	Single(
		sensors_components = vector_sensors, 
		vector_length = vector_length,
		nTimesteps = nTimesteps,
		name = 'Observer' if policy_type in ['MlpPolicy'] else 'VecObserver',
		)
	if policy_type not in ['MlpPolicy']:
		img_sensors = []
		img_sensors.append('DepthMap')
		Single(
			sensors_components = img_sensors, 
			is_image = True,
			image_height = 144, 
			image_width = 256,
			image_bands = 1,
			nTimesteps = nTimesteps,
			name = 'ImgObserver',
			)
		from observers.multi import Multi
		Multi(
			vector_observer_component = 'VecObserver',
			image_observer_component = 'ImgObserver',
			name = 'Observer',
			)

	## MODEL
		# we will use a TD3 algorithm from SB3
	from sb3models.dqn import DQN
	DQN(
		environment_component = 'Environment',
		policy = policy_type,
		buffer_size = buffer_size,
		device = device,
		name = 'Model',
	)

	# SPAWNER
		# moves drone to desired starting location
		# sets the target goal since it is typically dependent on the start location
	from spawners.random import Random
	Random(
		drone_component = 'Drone',
		roof_component = 'Rooftops', # checks spawn points for collisions
		bounds_component = 'MapBounds', # spawn within bounds
		goal_range = [0, 5], # start close to drone then we will move further with curriculum learning
		discretize = True, # spawns at integer values only
		name = 'Spawner',
		)

	## MODIFIERS
		# modifiers are like wrappers, and will add functionality before or after any component
	# CURRICULUM LEARNING
		# this modifier will be called at the end of every episode to see the percent of succesfull paths
		# if enough paths were succesfull then this will level up to harder goal
	from modifiers.curriculum import Curriculum
	Curriculum(
		base_component = 'Environment',
		parent_method = 'end',
		goal_component = 'Spawner', # which component to level up
		level_up_amount = [4, 4], # will increase min,max goal distance by these many meters
		level_up_criteria = 0.9, # percent of succesfull paths needed to level up
		level_up_buffer = 20, # number of previous episodes to look at to measure level_up_criteria
		max_level = 20, # can level up this many times after will terminate DRL learning loop
		order = 'post',
		frequency = 1, # check every this many episodes
		name = 'Curriculum',
	)
	# SAVERS
		# these will save any intermediate data we want during the training process
	from modifiers.saver import Saver
	checkpoint_freq = 100
	# save Train states and observations after each checkpoint
	Saver(
		base_component = 'Environment',
		parent_method = 'end',
		track_vars = [
					'observations', 
					'states',
					],
		write_folder = working_directory + 'states/',
		order = 'post',
		save_config = True,
		frequency = checkpoint_freq,
		name = 'EnvSaver',
	)
	# save model after each checkpoint
	Saver(
		base_component = 'Model',
		parent_method = 'end',
		track_vars = [
					'model', 
					#'replay_buffer', # this can cost alot of memory
					],
		write_folder = working_directory,
		order = 'post',
		frequency = checkpoint_freq,
		name = 'ModelSaver',
	)
	# if using horizontal movement, lets adjust to slight changes in altitude
	from modifiers.altadjust import AltAdjust
	AltAdjust(
		base_component = 'Actor',
		parent_method = 'step',
		drone_component = 'Drone',
		order = 'post',
		adjust = -4,
		name = 'AltAdjust',
	)


# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
utils.speak('learning loop terminated. Now exiting...')
configuration.controller.stop()
