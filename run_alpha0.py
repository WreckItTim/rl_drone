import utils
from configuration import Configuration
import math


# **** SETUP ****

# get OS, set file IO paths
run_name = 'gamma_alpha0' # subcategory of test type
OS = utils.setup(
	working_directory = 'local/runs/' + run_name + '/',
	)

# CREATE CONTROLLER
controller = utils.get_controller(
	controller_type = 'train',
	total_timesteps = 1_000, # optional if using train - all other hypers set from model instance
	continue_training = False, # if True will continue learning loop from last step saved, if False will reset learning loop
	model_component = 'Model', # if using train, set model
	environment_component = 'TrainEnvironment', # if using train, set train environment
	tb_log_name = 'run1', # logs tensor board to this directory
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
configuration = Configuration(meta, controller)


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
AirSimMap(
	voxels_component='Voxels',
	settings = {
		'ClockSpeed': 8,
		},
	setting_files = [
		'lightweight', 
		],
	release_path = 'local/airsim_maps/Blocks/Blocks',
	console_flags = [
		'-Windowed',
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
move_speed = 1 
yaw_rate = 11.25
step_duration = 2 
# ACTIONS
from actions.fixedmove import FixedMove 
FixedMove(
	drone_component = 'Drone', 
	x_speed = move_speed, 
	duration = step_duration,
	name = 'MoveForward',
)
FixedMove(
	drone_component = 'Drone', 
	x_speed = move_speed * 2, 
	duration = step_duration,
	name = 'MoveForward2',
)
from actions.fixedrotate import FixedRotate 
FixedRotate(
	drone_component = 'Drone',  
	yaw_rate = -1 * yaw_rate,
	duration = step_duration,
	name = 'RotateLeft',
)
FixedRotate(
	drone_component = 'Drone',  
	yaw_rate = -1 * yaw_rate * 2,
	duration = step_duration,
	name = 'RotateLeft2',
)
FixedRotate(
	drone_component = 'Drone',  
	yaw_rate = yaw_rate,
	duration = step_duration,
	name = 'RotateRight',
)
FixedRotate(
	drone_component = 'Drone',  
	yaw_rate = yaw_rate * 2,
	duration = step_duration,
	name = 'RotateRight2',
)
# ACTOR
from actors.discreteactor import DiscreteActor
DiscreteActor(
	actions_components=[
		'MoveForward',
		'MoveForward2',
		'RotateLeft',
		'RotateLeft2',
		'RotateRight',
		'RotateRight2',
		],
	name='Actor',
)


# CREATE MODEL
from models.dqn import DQN
DQN(
	environment_component = 'TrainEnvironment',
	policy = 'MlpPolicy',
	policy_kwargs = {'net_arch':[64,64]},
	buffer_size = 1000,
	learning_starts = 100,
	target_update_interval = 100,
	tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
	name='Model',
)


# CREATE GOAL
max_distance = 100
x_bounds = [-1*max_distance, max_distance]
y_bounds = [-1*max_distance, max_distance]
z_bounds = [-1*max_distance, -4]
from others.relativegoal import RelativeGoal
RelativeGoal(
	drone_component = 'Drone',
	map_component = 'Map',
	xyz_point = [6, 0, 0],
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
# OBSERVER
from observers.single import Single
Single(
	sensors_components = ['GoalDistance', 'GoalOrientation', 'FlattenedDepth'], 
	vector_length = 2 + len(flat_cols),
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
	include_z = False,
	name = 'GoalReward',
)
from rewards.distance import Distance
Distance(
	drone_component = 'Drone',
	goal_component = 'Goal',
	max_distance = max_distance,
	include_z = False,
	name = 'DistanceReward',
)
from rewards.steps import Steps
Steps(
	name = 'StepsReward',
)
# REWARDER
from rewarders.schema import Schema
Schema(
	rewards_components = [
		'CollisionReward',
		'GoalReward',
		'DistanceReward',
		'StepsReward',
	],
	reward_weights = [
		1,
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
			x_min=-100,
			x_max=100,
			y_min=-100,
			y_max=100,
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
from modifiers.evaluatorcharlie import EvaluatorCharlie
EvaluatorCharlie(
	base_component = 'TrainEnvironment',
	parent_method = 'reset',
	order='pre',
	evaluate_environment_component = 'EvaluateEnvironment',
	nEpisodes = 6,
	frequency = 100,
	name = 'Evaluator',
)
# SAVER
from modifiers.saver import Saver
Saver(
	base_component = 'TrainEnvironment',
	parent_method = 'reset',
	track_vars = [
				  'observations', 
				  'states',
				  ],
	order = 'post',
	frequency = 100,
	activate_on_first = False,
	name='TrainEnvSaver',
)
Saver(
	base_component = 'Model',
	parent_method = 'reset',
	track_vars = [
				  'model', 
				  'replay_buffer',
				  ],
	order = 'post',
	frequency = 100,
	on_evaluate = False,
	activate_on_first = False,
	name='ModelSaver',
)
Saver(
	base_component = 'EvaluateEnvironment',
	parent_method = 'reset',
	track_vars = [
				  'observations', 
				  'states',
				  ],
	order = 'post',
	frequency = 6,
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