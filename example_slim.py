import rl_utils as utils
from configuration import Configuration
import math

# path to airsim release file - precompiled binary
airsim_release_path = 'local/airsim_maps/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'

# render_screen=True will render AirSim graphics to screen using Unreal engine, 
	# set to false if running from SSH or want to save resources
render_screen = True

# set path to read rooftops from to determine collidable surface positions
rooftops_path = 'rooftops/Blocks.p' # match to map or use voxels if not available


# load pytorch models onto this device
device = 'cuda:0'

# set if we are slimming hidden layers (slim), input layer (sense), or both (fuse)
slim = True
sense = False
fuse = slim and sense

# turn on horizontal or vertical motion
vert_motion = False

# set path to pretrained navigation model weights
base_type = 'Navi'
if fuse:
	base_type += 'Fuse'
elif slim:
	base_type += 'Slim'
elif sense:
	base_type += 'Sense'
motion_type = 'vert' if vert_motion else 'horz'
navi_weights_path = 'pretrained/' + base_type + '_' + motion_type + '.pt'

# write run files to this directory
working_directory = 'local/runs/example_' + base_type.lower() + '/'

# number of previous timesteps to add to FIFO queue for observation space
nTimesteps = 4

# set this based on your memory, is for aux TD3
replay_buffer_size = 100_000

# setup for run, set system vars and prepare file system
utils.setup(working_directory, overwrite_directory=True) # WARNING: overwrite_directory will clear all old data in this folder

# make controller to run configuration on (we will train a model)
from controllers.train import Train
controller = Train(
	model_component = 'AuxModel',
	environment_component = 'AuxEnvironment',
	)

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

## create navigation environment component to handle step() and reset() for DRL model training
from environments.goalenv import GoalEnv
GoalEnv(
	drone_component = 'Drone', 
	actor_component = 'NaviActor', 
	observer_component = 'Observer', 
	rewarder_component = 'Rewarder',
	model_component = 'NaviModel',
	map_component = 'Map',
	spawner_component = 'Spawner',
	name = 'NaviEnvironment',
	)

## create auxiliary environment component to handle step() and reset() for DRL model training
from environments.auxenv import AuxEnv
AuxEnv(
		actor_component='AuxActor', 
		model_component='AuxModel',
		navi_component='NaviEnvironment',
		name='AuxEnvironment',
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
		'ClockSpeed': 8, # reduce this value if experiencing lag
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
reward_weights = []
# we will reward moving closer, reward reaching goal, penalize too many steps, and penalize collisions
rewards = []
# heavy penalty for collision
from rewards.collision import Collision
Collision(
	drone_component = 'Drone',
	name = 'CollisionReward',
	)
rewards.append('CollisionReward')
reward_weights.append(1)
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
reward_weights.append(10)
# heavy penalty for using too many steps
from rewards.maxsteps import MaxSteps
MaxSteps(
	name = 'MaxStepsReward',
	update_steps = True, # update the maximum number of steps based on distance to goal
	max_steps = 4, # base number of steps, will scale with further goal
	max_max = 50, # absolute max number of steps, regardless of scaling from further goals
	)
rewards.append('MaxStepsReward')
reward_weights.append(1)
# intermediate penalty for using more steps
from rewards.steps import Steps
Steps(
	name = 'StepsReward',
	)
rewards.append('StepsReward')
reward_weights.append(.1)
# intermediate reward for approaching goal
from rewards.distance import Distance
Distance(
	drone_component = 'Drone',
	goal_component = 'Spawner',
	include_z = True, # includes z in distance calculations
	name = 'DistanceReward',
	)
rewards.append('DistanceReward')
reward_weights.append(.1)
if slim:
	# penalize computational complexity
	from rewards.slim import Slim
	Slim(
		slim_component='SlimAction',
		name='SlimReward',
	)
	rewards.append('SlimReward')
	reward_weights.append(.1)
if sense:
	# penalty for higher resolutions=
	from rewards.resolution import Resolution
	Resolution(
		resolution_component = 'FlattenedDepthResolution',
		name = 'ResolutionReward',
	)
	rewards.append('ResolutionReward')
	reward_weights.append(.1)
# REWARDER
from rewarders.schema import Schema
Schema(
	rewards_components = rewards,
	reward_weights = reward_weights, 
	name = 'Rewarder',
	)

## ACTION SPACE

# navigation model actions
navi_actions = []
from actions.move import Move 
Move(
	drone_component = 'Drone', 
	base_x_rel = 10, # can move forward up to this many meters
	adjust_for_yaw = True, # this adjusts movement based on current yaw
	zero_thresh_abs = False,
	name = 'MoveForward',
	)
navi_actions.append('MoveForward')
Move(
	drone_component = 'Drone', 
	base_z_rel = 10, 
	adjust_for_yaw = True,
	active = vert_motion,
	name = 'MoveVertical',
)
navi_actions.append('MoveVertical')
from actions.rotate import Rotate 
Rotate(
	drone_component = 'Drone',  
	base_yaw = math.pi, # can rotate yaw by +/- this many radians
	name = 'Rotate',
	)
navi_actions.append('Rotate')
## ACTOR
from actors.teleportercontinuous import TeleporterContinuous
# we use a teleporter here because it is quicker and more stable
	# it will check collisions between current point and telported point then move directly to that location
TeleporterContinuous(
	drone_component = 'Drone',
	actions_components = navi_actions,
	name = 'NaviActor',
	)
	
# auxilary model actions
aux_actions = []	
if slim:
	from actions.slim import Slim
	Slim(
		model_component = 'NaviModel',
		name = 'SlimAction'
	) 
	aux_actions.append('SlimAction')
if sense:
	from actions.resolution import Resolution 
	Resolution(
		scales_components = [
			'ResizeFlat',
		],
		name = 'FlattenedDepthResolution',
	)
	aux_actions.append('FlattenedDepthResolution')
	if vert_motion:
		Resolution(
			scales_components = [
				'ResizeFlat2',
			],
			name = 'FlattenedDepthResolution2',
		)
		aux_actions.append('FlattenedDepthResolution2')
from actors.continuousactor import ContinuousActor
ContinuousActor(
	actions_components = aux_actions,
	name='AuxActor',
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
	min_input = 1,
	max_input = 125, # max depth
	left = 0,
	name = 'NormalizeDistance',
	)
from transformers.resizeimage import ResizeImage
image_shape=(25,25)
ResizeImage(
	image_shape=image_shape,
	name = 'ResizeImage',
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
# sense vert distance to goal
Distance(
	misc_component = 'Drone',
	misc2_component = 'Spawner',
	include_x = False,
	include_y = False,
	prefix = 'drone_to_goal',
	transformers_components = [
		'NormalizeDistance',
		],
	name = 'GoalAltitude',
)
# get flattened depth map (obsfucated front facing distance sensors)
from transformers.resizeflat import ResizeFlat
# airsim camera's image pixel size default is  "Width": 256, "Height": 144,
max_cols = [5*(i+1) for i in range(5)] # splits depth map by columns
max_rows = [5*(i+1) for i in range(5)] # splits depth map by rows
ResizeFlat(
	max_cols = max_cols,
	max_rows = max_rows,
	name = 'ResizeFlat',
	)
ResizeFlat(
	max_cols = max_cols,
	max_rows = max_rows,
	name = 'ResizeFlat2',
	)
from sensors.airsimcamera import AirSimCamera
AirSimCamera(
	airsim_component = 'Map',
	transformers_components = [
		'ResizeImage',
		'ResizeFlat',
		'NormalizeDistance',
		],
	name = 'FlattenedDepth',
	)
AirSimCamera(
	airsim_component = 'Map',
	camera_view='3', 
	transformers_components = [
		'ResizeImage',
		'ResizeFlat2',
		'NormalizeDistance',
		],
	name = 'FlattenedDepth2',
	)
# OBSERVER
# currently must count vector size of sensor output
vector_sensors = []
vector_length = 0
vector_sensors.append('FlattenedDepth')
vector_length +=  len(max_cols) * len(max_rows) # several more vector elements
vector_sensors.append('FlattenedDepth2')
vector_length +=  len(max_cols) * len(max_rows) # several more vector elements
vector_sensors.append('GoalDistance')
vector_length +=  1
vector_sensors.append('GoalOrientation')
vector_length +=  1
vector_sensors.append('GoalAltitude')
vector_length +=  1
from observers.single import Single
Single(
	sensors_components = vector_sensors, 
	vector_length = vector_length,
	nTimesteps = nTimesteps,
	name = 'Observer',
	)

## MODEL
from sb3models.td3 import TD3
# navigation model (read pretrained from file)
TD3(
	environment_component = 'NaviEnvironment',
	policy = 'MlpPolicy',
	policy_kwargs = {'net_arch':[64, 32, 32]},
	buffer_size = 1,
	read_weights_path = navi_weights_path,
	convert_slim = slim,
	use_slim = slim,
	device = device,
	name = 'NaviModel',
)
# auxiliary model (to be trained)
TD3(
	environment_component = 'AuxEnvironment',
	policy = 'MlpPolicy',
	policy_kwargs = {'net_arch':[32,32]},
	buffer_size = replay_buffer_size,
	device = device,
	name = 'AuxModel',
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
	name = 'Spawner',
	)

## MODIFIERS
	# modifiers are like wrappers, and will add functionality before or after any component
# CURRICULUM LEARNING
	# this modifier will be called at the end of every episode to see the percent of succesfull paths
	# if enough paths were succesfull then this will level up to harder goal
from modifiers.curriculum import Curriculum
Curriculum(
	base_component = 'NaviEnvironment',
	parent_method = 'end',
	goal_component = 'Spawner', # which component to level up
	level_up_amount = [5, 5], # will increase min,max goal distance by these many meters
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
	base_component = 'NaviEnvironment',
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
	base_component = 'AuxModel',
	parent_method = 'end',
	track_vars = [
				'model', 
				'replay_buffer', # this can cost alot of memory # TODO: comment out
				],
	write_folder = 'local/airsim_maps/local/', # TODO: change back to working_directory
	order = 'post',
	frequency = checkpoint_freq,
	name = 'ModelSaver',
)
if not vert_motion:
	# if using horizontal movement, lets adjust to slight changes in altitude
	from modifiers.altadjust import AltAdjust
	AltAdjust(
		base_component = 'NaviActor',
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
