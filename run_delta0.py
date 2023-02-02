import utils
from configuration import Configuration
import math


# **** SETUP ****

# get OS, set file IO paths
run_name = 'gamma2_delta0_hackfest4_run3' # subcategory of test type
OS = utils.setup(
	working_directory = 'local/runs/' + run_name + '/',
	)

# CREATE CONTROLLER
continue_training = False
read_model_path = None
read_replay_buffer_path = None
if continue_training:
	read_model_path = run_name + '/Model/model.zip'
	read_replay_buffer_path = run_name + '/Model/replay_buffer.zip'
controller = utils.get_controller(
	controller_type = 'train',
	total_timesteps = 1_000_000, # optional if using train - all other hypers set from model instance
	continue_training = continue_training, # if True will continue learning loop from last step saved, if False will reset learning loop
	model_component = 'Model', # if using train, set model
	environment_component = 'TrainEnvironment', # if using train, set train environment
	tb_log_name = 'tb_log', # logs tensor board to this directory
	)

# CREATE CONFIGURATION
# will add components to this configuration automatically
# can switch which configuration is active to add to different once
# I almost always just use 1 configuration per run
# SET META DATA (anything you want here, just writes to config file as a dict)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'repo_version': 'gamma_alpha',
	'run_name': run_name,
	'timestamp': utils.get_timestamp(),
	'run_OS': utils.get_global_parameter('OS'),
	'absolute_path' : utils.get_global_parameter('absolute_path'),
	'working_directory' : utils.get_global_parameter('working_directory'),
	}
configuration = Configuration(
	meta, 
	controller, 
	add_timers=False, 
	add_memories=False,
	)


# **** CREATE COMPONENTS ****


# CREATE TRAIN ENVIRONMENT
from environments.goalenv import GoalEnv
GoalEnv(
	drone_component='Drone', 
	actor_component='Actor', 
	observer_component='Observer', 
	rewarder_component='Rewarder', 
	goal_component='Goal',
	overide_timer=True, # time benchmark on
	name='TrainEnvironment',
)
# CREATE EVALUATE ENVIRONMENT
GoalEnv(
	drone_component='Drone', 
	actor_component='Actor', 
	observer_component='Observer', 
	rewarder_component='Rewarder', 
	goal_component='Goal',
	is_evaluation_env=True,
	name='EvaluateEnvironment',
)


# CREATE MAP
from maps.airsimmap import AirSimMap
release_path = None
if utils.get_global_parameter('OS') == 'windows':
	release_path = 'local/airsim_maps/Blocks/WindowsNoEditor/Blocks.exe'
if utils.get_global_parameter('OS') == 'linux':
	release_path = 'local/airsim_maps/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
AirSimMap(
	voxels_component='Voxels',
	release_path = release_path,
	settings = {
		'ClockSpeed': 8,
		},
	setting_files = [
		'lightweight', 
		],
	console_flags = [
		'-Windowed',
		#'-RenderOffscreen',

	],
	name = 'Map',
)
from others.voxels import Voxels
Voxels(
	relative_path = utils.get_global_parameter('working_directory') + 'map_voxels.binvox',
	map_component = 'Map',
	name = 'Voxels',
	)


# CREATE DRONE
from drones.airsimdrone import AirSimDrone
AirSimDrone(
	airsim_component = 'Map',
	name='Drone',
)


# CREATE ACTION SPACE
base_move_speed = 4
base_yaw_rate = 90
step_duration = 2 
from actions.move import Move 
Move(
	drone_component = 'Drone', 
	base_x_speed = base_move_speed, 
	duration = step_duration,
	zero_min_threshold=-10,
	zero_max_threshold=1/4,
	name = 'MoveForward',
)
Move(
	drone_component = 'Drone', 
	base_z_speed = base_move_speed, 
	duration = step_duration,
	zero_min_threshold=-1/4,
	zero_max_threshold=1/4,
	name = 'MoveVertical',
)
from actions.rotate import Rotate 
Rotate(
	drone_component = 'Drone',  
	base_yaw_rate = base_yaw_rate,
	duration = step_duration,
	zero_min_threshold=-1/6,
	zero_max_threshold=1/6,
	name = 'Rotate',
)
# ACTOR
actions=[
	'MoveForward',
	'MoveVertical',
	'Rotate',
	]
from actors.continuousactor import ContinuousActor
ContinuousActor(
	actions_components = actions,
	name='Actor',
)


# CREATE MODEL
from models.td3 import TD3
TD3(
	environment_component = 'TrainEnvironment',
	policy = 'MlpPolicy',
	policy_kwargs = {'net_arch':[64,64]},
	buffer_size = 1000,
	learning_starts = 100,
	read_model_path = read_model_path,
	read_replay_buffer_path = read_replay_buffer_path,
	tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
	overide_memory = True, # memory benchmark on
	name='Model',
)


# CREATE GOAL
max_distance = 100
x_bounds = [-1*max_distance, max_distance]
y_bounds = [-1*max_distance, max_distance]
z_bounds = [-20, -4]
from others.relativegoal import RelativeGoal
RelativeGoal(
	drone_component = 'Drone',
	map_component = 'Map',
	xyz_point = [6, 6, 0],
	random_point_on_train = True,
	random_point_on_evaluate = False,
	random_dim_min = 6,
	random_dim_max = 10,
	x_bounds = x_bounds,
	y_bounds = y_bounds,
	z_bounds = z_bounds,
	random_yaw_on_train = True,
	random_yaw_on_evaluate = False,
	random_yaw_min = -1 * math.pi,
	random_yaw_max = math.pi,
	name = 'Goal',
	)


# CREATE OBSERVATION SPACE
flat_cols = [16, 32, 52, 68, 84]
flat_rows = [21, 42, 63, 84]
# OBSERVER
from observers.single import Single
Single(
	sensors_components = ['GoalDistance', 'GoalOrientation', 'GoalAltitude', 'FlattenedDepth', 'Moves'], 
	vector_length = 1 + 1 + 1 + len(flat_cols)*len(flat_rows) + len(actions),
	nTimesteps = 4,
	name = 'Observer',
)
# SENSORS
from sensors.distance import Distance
Distance(
	misc_component = 'Drone',
	misc2_component = 'Goal',
	include_z = False,
	prefix = 'drone_to_goal',
	transformers_components = [
		'PositionNoise',
		'NormalizeDistance',
		], 
	name = 'GoalDistance',
)
from sensors.orientation import Orientation
Orientation(
	misc_component = 'Drone',
	misc2_component = 'Goal',
	prefix = 'drone_to_goal',
	transformers_components = [
		'OrientationNoise',
		'NormalizeOrientation',
		],
	name = 'GoalOrientation',
)
Distance(
	misc_component = 'Drone',
	misc2_component = 'Goal',
	include_x = False,
	include_y = False,
	prefix = 'drone_to_goal',
	transformers_components = [
		'PositionNoise',
		'NormalizeDistance',
		],
	name = 'GoalAltitude',
)
from sensors.airsimcamera import AirSimCamera
AirSimCamera(
	airsim_component = 'Map',
	transformers_components = [
		'ResizeImage',
		'DepthNoise',
		'NormalizeDistance',
		'ResizeFlat',
		],
	name = 'FlattenedDepth',
	)
from sensors.moves import Moves
Moves(
	actor_component = 'Actor',
	name = 'Moves',
	)
# TRANSFORMERS
from transformers.gaussiannoise import GaussianNoise
GaussianNoise(
	name = 'PositionNoise',
)
GaussianNoise(
	deviation = math.radians(4),
	name = 'OrientationNoise',
)
from transformers.gaussianblur import GaussianBlur
GaussianBlur(
	name = 'DepthNoise',
)
from transformers.normalize import Normalize
Normalize(
	min_input = -1 * math.pi, # min angle
	max_input = math.pi, # max angle
	name = 'NormalizeOrientation',
)
Normalize(
	max_input = max_distance, # max depth
	name = 'NormalizeDistance',
)
from transformers.resizeimage import ResizeImage
ResizeImage(
	name = 'ResizeImage',
)
from transformers.resizeflat import ResizeFlat
ResizeFlat(
	max_cols = flat_cols,
	max_rows = flat_rows,
	name = 'ResizeFlat',
)


# CREATE REWARDS AND TERMINATORS
# REWARDS
from rewards.collision import Collision
Collision(
	drone_component = 'Drone',
	name = 'CollisionReward',
)
from rewards.goal import Goal
Goal(
	drone_component = 'Drone',
	goal_component = 'Goal',
	max_distance = max_distance,
	include_z = True,
	name = 'GoalReward',
)
from rewards.steps import Steps
Steps(
	max_steps=16,
	step_ratio=1,
	name = 'StepsReward',
)
# REWARDER
from rewarders.schema import Schema
Schema(
	rewards_components = [
		'CollisionReward',
		'GoalReward',
		'StepsReward',
	],
	reward_weights = [
		1,
		1,
		1,
	],
	name = 'Rewarder',
)


# CREATE MODIFIERS
# SPAWNER
start_z = -4 
from modifiers.spawner import Spawner
from others.spawn import Spawn
Spawner(
	base_component = 'Drone',
	parent_method = 'reset',
	drone_component = 'Drone',
	spawns_components=[
		Spawn(
			map_component = 'Map',
			x_min=-1*max_distance,
			x_max=max_distance,
			y_min=-1*max_distance,
			y_max=max_distance,
			z_min=start_z,
			z_max=start_z,
			yaw_min = -1 * math.pi,
			yaw_max = math.pi,
			random=True,
		),
	],
	order='post',
	on_evaluate = False,
	name='TrainSpawner',
)
Spawner(
	base_component = 'Drone',
	parent_method = 'reset',
	drone_component = 'Drone',
	spawns_components=[
		Spawn(
			z=start_z,
			yaw=math.radians(0),
			),
		Spawn(
			z=start_z,
			yaw=math.radians(45),
			),
		Spawn(
			z=start_z,
			yaw=math.radians(135),
			),
		Spawn(
			z=start_z,
			yaw=math.radians(180),
			),
		Spawn(
			z=start_z,
			yaw=math.radians(-130),
			),
		Spawn(
			z=start_z,
			yaw=math.radians(-45),
			),
	],
	order='post',
	on_train = False,
	name='EvaluateSpawner',
)
# EVALUATOR
checkpoint = 100
nEvalEpisodes = 6
from modifiers.evaluatorcharlie import EvaluatorCharlie
EvaluatorCharlie(
	base_component = 'TrainEnvironment',
	parent_method = 'end',
	order = 'post',
	evaluate_environment_component = 'EvaluateEnvironment',
	model_component = 'Model',
	nEpisodes = nEvalEpisodes,
	frequency = checkpoint,
	activate_on_first = False,
	verbose = 1,
	name = 'Evaluator',
)
# SAVERS
from modifiers.saver import Saver
Saver(
	base_component = 'TrainEnvironment',
	parent_method = 'end',
	track_vars = [
				  'observations', 
				  'states',
				  ],
	order = 'pre',
	save_config = True,
	save_benchmarks = True,
	frequency = checkpoint,
	activate_on_first = False,
	name='TrainEnvSaver',
)
Saver(
	base_component = 'Model',
	parent_method = 'end',
	track_vars = [
				  'model', 
				  'replay_buffer',
				  ],
	order = 'pre',
	frequency = nEvalEpisodes,
	activate_on_first = False,
	name='ModelSaver',
)
Saver(
	base_component = 'EvaluateEnvironment',
	parent_method = 'end',
	track_vars = [
				  'observations', 
				  'states',
				  ],
	order = 'pre',
	frequency = nEvalEpisodes,
	activate_on_first = False,
	name='EvalEnvSaver',
)


utils.speak('configuration created!')


stopwatch = utils.StopWatch()
# CONNECT COMPONENTS
model = str(configuration.get_component('Model')._child())
configuration.connect_all()
if 'dqn' in model:
	print(configuration.get_component('Model')._sb3model.q_net)
if 'ddpg' in model or 'td3'  in model:
	print(configuration.get_component('Model')._sb3model.critic)
utils.speak('all components connected. Send any key to continue...')
x = input()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()
stopwatch.stop()
utils.speak(f'ran in {stopwatch.delta_time / 3600} hours')

# done
configuration.controller.stop()
