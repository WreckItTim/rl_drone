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
	terminators_components=[
		'CollisionTerminator',
		'GoalTerminator',
		'StepsTerminator',
		'DistanceTerminator',
		],
	goal_component='Goal',
	overide_timer = True, # time benchmark on
	name = 'TrainEnvironment',
)
# CREATE EVALUATE ENVIRONMENT
GoalEnv(
	drone_component='Drone', 
	actor_component='Actor', 
	observer_component='Observer', 
	rewarder_component='Rewarder', 
	terminators_components=[
		'CollisionTerminator',
		'GoalTerminator',
		'StepsTerminator',
		'DistanceTerminator',
		],
	goal_component='Goal',
	is_evaluation_environment=True,
	name = 'EvaluateEnvironment',
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




from other.relativegoal import RelativeGoal
RelativeGoal(
	drone_component = 'Drone',
	map_component = 'Map',
	xyz_point = [6, 0, 0],
	random_point_on_train = True,
	random_point_on_evaluate = False,
	min_amp_up = 0,
	max_amp_up = 0,
	random_dim_min = 6,
	random_dim_max = 10,
	x_bounds = x_bounds,
	y_bounds = y_bounds,
	z_bounds = z_bounds,
	random_yaw_on_train = False,
	random_yaw_on_evaluate = False,
	random_yaw_min = -1 * math.pi,
	random_yaw_max = math.pi,
	reset_on_step = False,
	name = 'Goal',
	)

flat_cols = [16, 32, 52, 68, 84]
flat_rows = [21, 42, 63, 84]
other_length = 3
vector_length = len(flat_cols)*len(flat_rows) + other_length
# vector shape is hard coded
# set number of timesteps to keep in current state
nTimesteps = 4
# set modality Multi used
observation = 'Vector' # Image Vector Multi
# set observer component to handle the observation space
observer = 'Multi' if observation == 'Multi' else 'Single'
# detrmine to include z-axis (vertical) in objective during calulations
include_z = True
# voxels to check valid spawn/objective points on map and visualize results (optional)
use_voxels = True
# set max steps
max_steps = 14
# set tolerance to reach goal within (arbitrary units depending on drone)
goal_tolerance = 4
# set action space type
action_type = 'discrete' # discrete continuous
# how many episodes in each evaluation set?
num_eval_episodes = 6
# how many training episode before we evaluate/update?
evaluate_frequency = 100
# bounds on map (where the drone can go)
max_distance = 100
x_bounds = [-1*max_distance, max_distance]
y_bounds = [-1*max_distance, max_distance]
z_bounds = [-1*max_distance, -4]
	



# TRANSFORMER, 0 reserved for bad data
from transformers.gaussiannoise import GaussianNoise
GaussianNoise(
	mean = 0,
	deviation = 0.5,
	name = 'PositionNoise',
)
GaussianNoise(
	mean = 0,
	deviation = math.radians(4),
	name = 'OrientationNoise',
)
from transformers.gaussianblur import GaussianBlur
GaussianBlur(
	sigma = 2,
	name = 'DepthNoise',
)
from transformers.normalize import Normalize
Normalize(
	min_input = -1*max_distance, # min distance
	max_input = max_distance, # max distance
	min_output = 0.1, # SB3 uses 0-1 floating point values
	max_output = 1, # SB3 uses 0-1 floating point values
	name = 'NormalizePosition',
)
Normalize(
	min_input = -1 * math.pi, # min angle
	max_input = math.pi, # max angle
	min_output = 0.1, # SB3 uses 0-1 floating point values
	max_output = 1, # SB3 uses 0-1 floating point values
	name = 'NormalizeOrientation',
)
Normalize(
	min_input = 0, # min depth
	max_input = max_distance, # max depth
	min_output = 1, # SB3 uses 0-255 pixel values
	max_output = 255, # SB3 uses 0-255 pixel values
	name = 'NormalizeDepth',
)
Normalize(
	min_input = 0, # min depth
	max_input = max_distance, # max depth
	min_output = 0, # SB3 uses 0-1 floating point values
	max_output = 1, # SB3 uses 0-1 floating point values
	name = 'NormalizeDistance',
)
from transformers.resizeimage import ResizeImage
ResizeImage(
	image_shape = (image_height, image_width),
	name = 'ResizeImage',
)
from transformers.resizeflat import ResizeFlat
ResizeFlat(
	max_cols = flat_cols,
	max_rows = flat_rows,
	name = 'ResizeFlat',
)

# SENSOR
if drone == 'AirSim' and 'Camera' in image_sensors:
	# images are 256 x 144 (width x height)
	from sensors.airsimcamera import AirSimCamera
	AirSimCamera(
		airsim_component = 'Map',
		raw_code = 'AirSimDepthCamera',
		camera_view = '0',
		image_type = 2,
		as_float = True,
		compress = False,
		is_gray = True,
		transformers_components = [
			'ResizeImage',
			'DepthNoise',
			'NormalizeDepth',
			],
		name = 'Camera',
		)
if drone == 'AirSim' and 'Distance' in vector_sensors:
	from sensors.airsimdistance import AirSimDistance
	AirSimDistance(
		airsim_component = 'Map',
		transformers_components = [
			'PositionNoise',
			'NormalizeDistance',
			],
		name = 'Distance',
	)
if drone == 'Tello' and 'Camera' in image_sensors:
	from sensors.portcamera import PortCamera
	PortCamera(
		port = 'udp://0.0.0.0:11111',
		is_gray = False,
		transformers_components = [
			'ResizeImage',
			'DepthNoise',
			'NormalizeDepth',
			],
		name = 'Camera',
	)
if 'DronePosition' in vector_sensors:
	from sensors.position import Position
	Position(
		misc_component = 'Drone',
		prefix = 'drone',
		transformers_components = [
			'PositionNoise',
			'NormalizePosition',
			],
		name = 'DronePosition',
	)
if 'DroneOrientation' in vector_sensors:
	from sensors.orientation import Orientation
	Orientation(
		misc_component = 'Drone',
		prefix = 'drone',
		transformers_components = [
			'OrientationNoise',
			'NormalizeOrientation',
			],
		name = 'DroneOrientation',
	)
if 'GoalPosition' in vector_sensors:
	from sensors.position import Position
	Position(
		misc_component = 'Goal',
		prefix = 'goal',
		transformers_components = [
			'PositionNoise',
			'NormalizePosition',
			],
		name = 'GoalPosition',
	)
if 'GoalOrientation' in vector_sensors:
	from sensors.orientation import Orientation
	Orientation(
		misc_component = 'Goal',
		prefix = 'goal',
		transformers_components = [
			'OrientationNoise',
			'NormalizeOrientation',
			],
		name = 'GoalOrientation',
	)
if 'DroneToGoalPosition' in vector_sensors:
	from sensors.position import Position
	Position(
		misc_component = 'Drone',
		misc2_component = 'Goal',
		prefix = 'drone_to_goal',
		transformers_components = [
			'PositionNoise',
			'NormalizePosition',
			], 
		name = 'DroneToGoalPosition',
	)
if 'DroneToGoalDistance' in vector_sensors:
	from sensors.distance import Distance
	Distance(
		misc_component = 'Drone',
		misc2_component = 'Goal',
		include_x = True,
		include_y = True,
		include_z = False,
		prefix = 'drone_to_goal',
		transformers_components = [
			'PositionNoise',
			'NormalizeDistance',
			], 
		name = 'DroneToGoalDistance',
	)
if 'DroneToGoalOrientation' in vector_sensors:
	from sensors.orientation import Orientation
	Orientation(
		misc_component = 'Drone',
		misc2_component = 'Goal',
		prefix = 'drone_to_goal',
		transformers_components = [
			'OrientationNoise',
			'NormalizeOrientation',
			],
		name = 'DroneToGoalOrientation',
	)
if 'DroneToGoalAltitude' in vector_sensors:
	from sensors.distance import Distance
	Distance(
		misc_component = 'Drone',
		misc2_component = 'Goal',
		include_x = False,
		include_y = False,
		include_z = True,
		prefix = 'drone_to_goal',
		transformers_components = [
			'PositionNoise',
			'NormalizeDistance',
			],
		name = 'DroneToGoalAltitude',
	)
if 'FlattenedCamera' in vector_sensors:
	from sensors.airsimcamera import AirSimCamera
	AirSimCamera(
		airsim_component = 'Map',
		raw_code = 'AirSimDepthCamera',
		camera_view = '0',
		image_type = 2,
		as_float = True,
		compress = False,
		is_gray = True,
		transformers_components = [
			'ResizeImage',
			'DepthNoise',
			'NormalizeDistance',
			'ResizeFlat',
			],
		name = 'FlattenedCamera',
		)
	
# OBSERVER
if observer == 'Single':
	from observers.single import Single
	if observation == 'Vector':
		sensor_array = vector_sensors
	if observation == 'Image':
		sensor_array = image_sensors
	Single(
		sensors_components = sensor_array, 
		vector_length = vector_length,
		is_image = observation == 'Image',
		image_height = image_height, 
		image_width = image_width,
		image_bands = image_bands,
		nTimesteps = nTimesteps,
		name = 'Observer',
	)
if observer == 'Multi':
	from observers.single import Single
	Single(
		sensors_components = vector_sensors, 
		vector_length = vector_length,
		is_image = False,
		nTimesteps = nTimesteps,
		name = 'ObserverVector',
	)
	Single(
		sensors_components = image_sensors, 
		is_image = True,
		image_height = image_height, 
		image_width = image_width,
		image_bands = image_bands,
		nTimesteps = nTimesteps,
		name = 'ObserverImage',
	)
	from observers.multi import Multi
	Multi(
		vector_observer_component = 'ObserverVector',
		image_observer_component = 'ObserverImage',
		name = 'Observer',
		)


# REWARD
from rewards.avoid import Avoid
Avoid(
	drone_component = 'Drone',
	name = 'AvoidReward',
)
from rewards.goal import Goal
Goal(
	drone_component = 'Drone',
	goal_component = 'Goal',
	tolerance = goal_tolerance, 
	include_z = include_z,
	to_start=True,
	name = 'GoalReward',
)
from rewards.distance import Distance
Distance(
	drone_component = 'Drone',
	goal_component = 'Goal',
	max_distance = max_distance,
	include_z = include_z,
	name = 'DistanceReward',
)
from rewards.steps import Steps
Steps(
	max_steps = max_steps,
	name = 'StepsReward',
)
from rewards.bounds import Bounds
Bounds(
	drone_component = 'Drone',
	x_bounds = x_bounds,
	y_bounds = y_bounds,
	z_bounds = z_bounds,
	name = 'BoundsReward',
)
# REWARDER
from rewarders.schema import Schema
Schema(
	rewards_components = [
		'AvoidReward',
		'GoalReward',
		'StepsReward',
		'DistanceReward',
		#'BoundsReward',
	],
	reward_weights = [
		1,
		1,
		1,
		1,
		#1,
	],
	name = 'Rewarder',
)

# TERMINATOR
from terminators.collision import Collision
Collision(
	drone_component = 'Drone',
	name = 'CollisionTerminator',
)
from terminators.goal import Goal
Goal(
	drone_component = 'Drone',
	goal_component = 'Goal',
	tolerance = goal_tolerance, 
	include_z = include_z,
	name = 'GoalTerminator',
)
from terminators.distance import Distance
Distance(
	drone_component = 'Drone',
	goal_component = 'Goal',
	max_distance = max_distance,
	include_z = include_z,
	name = 'DistanceTerminator',
)
from terminators.rewardthresh import RewardThresh
RewardThresh(
	rewarder_component='Rewarder',
	min_reward = 0,
	name = 'RewardTerminator',
)
from terminators.maxsteps import MaxSteps
MaxSteps(
	max_steps = max_steps,
	name = 'StepsTerminator',
)
from terminators.bounds import Bounds
Bounds(
	drone_component = 'Drone',
	x_bounds = x_bounds,
	y_bounds = y_bounds,
	z_bounds = z_bounds,
	name = 'BoundsTerminator',
)

# MODEL
print('observation', observation)
policy_kwargs = None
if observation == 'Image': 
	policy = 'CnnPolicy'
elif observation == 'Vector': 
	policy = 'MlpPolicy'
elif observation == 'Multi': 
	policy = 'MultiInputPolicy'
print('policy', policy)
policy_kwargs = {
	'net_arch':[64,64],
}
if model == 'Hyper':
	from models.hyper import Hyper
	Hyper(
		environment_component = 'TrainEnvironment',
		_space = {
			'learning_rate':hp.quniform('learning_rate', 1, 6, 1),
			'tau':hp.uniform('tau', 0, 1.0),
			'gamma':hp.quniform('gamma', 1, 6, 1),
		},
		model_type = 'DQN',
		default_params={
			'policy': policy,
			'policy_kwargs': policy_kwargs,
			'verbose': 0,
			'buffer_size': evaluate_frequency * 10,
			'learning_starts': evaluate_frequency
		},
		nRuns = 4,
		max_evals = 32,
		name='Model',
	)
elif model == 'DQN':
	from models.dqn import DQN
	DQN(
		environment_component = 'TrainEnvironment',
		policy = policy,
		learning_rate = 1e-6,
		buffer_size = evaluate_frequency * 10,
		learning_starts = evaluate_frequency,
		batch_size = 32,
		tau = .8428,
		gamma = 0.9999,
		train_freq = 4,
		gradient_steps = 1,
		replay_buffer_class = None,
		replay_buffer_kwargs = None,
		optimize_memory_usage = False,
		target_update_interval = evaluate_frequency * 4,
		exploration_fraction = 0.1,
		exploration_initial_eps = 1.0,
		exploration_final_eps = 0.05,
		max_grad_norm = 1,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		create_eval_env = False,
		policy_kwargs = policy_kwargs,
		verbose = 0,
		seed = None,
		device = "auto",
		init_setup_model = True,
		model_path = read_model_path if read_model else None,
		replay_buffer_path = read_replay_buffer_path if read_replay_buffer else None,
		name='Model',
	)
elif model == 'TD3':
	from models.td3 import TD3
	TD3(
		environment_component = 'TrainEnvironment',
		policy = policy,
		learning_rate = 0.01,
		buffer_size = evaluate_frequency * 10,
		learning_starts = evaluate_frequency,
		batch_size = 100,
		tau = 0.1593,
		gamma = 0.99999,
		train_freq = (1, "episode"),
		gradient_steps = -1,
		action_noise = None,
		replay_buffer_class = None,
		replay_buffer_kwargs = None,
		optimize_memory_usage = False,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		create_eval_env = False,
		policy_kwargs = policy_kwargs,
		verbose = 0,
		seed = None,
		device = "auto",
		init_setup_model = True,
		model_path = read_model_path if read_model else None,
		replay_buffer_path = read_replay_buffer_path if read_replay_buffer else None,
		name='Model',
	)
'''
elif model == 'DDPG':
	from models.ddpg import DDPG
	DDPG(
		environment_component = 'TrainEnvironment',
		policy = policy,
		learning_rate = 1e-3,
		buffer_size = evaluate_frequency * 10,
		learning_starts = evaluate_frequency,
		batch_size = 100,
		tau = 0.005,
		gamma = 0.99,
		train_freq = (1, "episode"),
		gradient_steps = -1,
		action_noise = None,
		replay_buffer_class = None,
		replay_buffer_kwargs = None,
		optimize_memory_usage = False,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		create_eval_env = False,
		policy_kwargs = None,
		verbose = 0,
		seed = None,
		device = "auto",
		init_setup_model = True,
		model_path = read_model_path if read_model else None,
		replay_buffer_path = read_replay_buffer_path if read_replay_buffer else None,
		name='Model',
	)
elif model == 'A2C':
	from models.a2c import A2C
	A2C(
		environment_component = 'TrainEnvironment',
		policy = policy,
		learning_rate = 7e-4,
		n_steps = 5,
		gamma = 0.99,
		gae_lambda = 1.0,
		ent_coef = 0.0,
		vf_coef = 0.5,
		max_grad_norm = 0.5,
		rms_prop_eps = 1e-5,
		use_rms_prop = True,
		use_sde = False,
		sde_sample_freq = -1,
		normalize_advantage = False,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		create_eval_env = False,
		policy_kwargs = None,
		verbose = 0,
		seed = None,
		device = "auto",
		init_setup_model = True,
		model_path = read_model_path if read_model else None,
		replay_buffer_path = read_replay_buffer_path if read_replay_buffer else None,
		name='Model',
	)
elif model == 'PPO':
	from models.ppo import PPO
	PPO(
		environment_component = 'TrainEnvironment',
		policy = policy,
		learning_rate = 1e-3,
		n_steps = 2048,
		batch_size = 64,
		n_epochs = 10,
		gamma = 0.99,
		gae_lambda = 0.95,
		clip_range = 0.2,
		clip_range_vf = None,
		normalize_advantage = True,
		ent_coef = 0.0,
		vf_coef = 0.5,
		max_grad_norm = 0.5,
		use_sde = False,
		sde_sample_freq = -1,
		target_kl = None,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		create_eval_env = False,
		policy_kwargs = None,
		verbose = 0,
		seed = None,
		device = "auto",
		init_setup_model = True,
		model_path = read_model_path if read_model else None,
		replay_buffer_path = read_replay_buffer_path if read_replay_buffer else None,
		name='Model',
	)
elif model == 'SAC':
	from models.sac import SAC
	SAC(
		environment_component = 'TrainEnvironment',
		policy = policy,
		learning_rate = 1e-3,
		buffer_size = every_nEpisodes * 10,
		learning_starts = every_nEpisodes,
		batch_size = 256,
		tau = 0.005,
		gamma = 0.99,
		train_freq = 1,
		gradient_steps = 1,
		action_noise = None,
		replay_buffer_class = None,
		replay_buffer_kwargs = None,
		optimize_memory_usage = False,
		ent_coef = "auto",
		target_update_interval = 1,
		target_entropy = "auto",
		use_sde = False,
		sde_sample_freq = -1,
		use_sde_at_warmup = False,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		create_eval_env = False,
		policy_kwargs = None,
		verbose = 0,
		seed = None,
		device = "auto",
		init_setup_model = True,
		model_path = read_model_path if read_model else None,
		replay_buffer_path = read_replay_buffer_path if read_replay_buffer else None,
		name='Model',
	)
'''
	

# SPAWNER
start_z = -4 
from others.spawner import Spawner
from datastructs.spawn import Spawn
Spawner(
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
	name='TrainSpawner',
)
Spawner(
	spawns_components=[
		Spawn(
			z=start_z,
			yaw=math.radians(0),
			random=False,
			),
		Spawn(
			z=start_z,
			yaw=math.radians(45),
			random=False,
			),
		Spawn(
			z=start_z,
			yaw=math.radians(135),
			random=False,
			),
		Spawn(
			z=start_z,
			yaw=math.radians(180),
			random=False,
			),
		Spawn(
			z=start_z,
			yaw=math.radians(-130),
			random=False,
			),
		Spawn(
			z=start_z,
			yaw=math.radians(-45),
			random=False,
			),
	],
	name='EvaluateSpawner',
)

# EVALUATOR
from others.evaluator import Evaluator
Evaluator(
	train_environment_component = 'TrainEnvironment',
	evaluate_environment_component = 'EvaluateEnvironment',
	model_component = 'Model',
	frequency = evaluate_frequency,
	nEpisodes = num_eval_episodes,
	stopping_reward = 10,
	curriculum = True,
	goal_component = 'Goal',
	steps_components = [
		'StepsReward',
		'StepsTerminator',
	],

	name = 'Evaluator',
)

# SAVER
from others.saver import Saver
Saver(
	environment_component = 'TrainEnvironment',
	save_components = [
		'TrainEnvironment',
		'Model',
	],
	save_variables = {
		'TrainEnvironment':[
			'states',
			'observations',
		],
		'Model':[
			'model',
			'replay_buffer',
		],
	},
	frequency=evaluate_frequency, 
	save_configuration_file=True,
	save_benchmarks=True,
	name='TrainSaver',
)
from others.saver import Saver
Saver(
	environment_component = 'EvaluateEnvironment',
	save_components = [
		'EvaluateEnvironment',
	],
	save_variables = {
		'EvaluateEnvironment':[
			'states',
			'observations',
		],
	},
	frequency=num_eval_episodes, 
	save_configuration_file=False,
	save_benchmarks=True,
	name='EvaluateSaver',
)

utils.speak('configuration created!')


t1 = time()
# CONNECT COMPONENTS
model = str(configuration.get_component('Model')._child())
print('connecting with model', model)
configuration.connect_all()
print('connected with model', model)
if 'dqn' in model:
	print(configuration.get_component('Model')._sb3model.q_net)
if 'ddpg' in model or 'td3'  in model:
	print(configuration.get_component('Model')._sb3model.critic)
print('all components connected. Send any key to continue...')
x = input()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()
t2 = time()
delta_t = (t2 - t1) / 3600
print('ran in', delta_t, 'hours')


# done
configuration.controller.stop()
