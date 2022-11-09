import utils
from configuration import Configuration
import math
import numpy as np
from time import time
from hyperopt import hp

# GET OS
utils.set_operating_system()


# CREATE and set read/write DIRECTORIES
test_name = 'omega1' # subcategory of test type
working_directory = 'local/runs/' + test_name + '/'
utils.set_read_write_paths(working_directory = working_directory)


# SET META DATA (anything you want here, just write to config file)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'timestamp': utils.get_timestamp(),
	'repo_version': 'lambda',
	'test_name': test_name,
	}
# select rather to overwrite meta data in configuration file (if already exists)
update_meta = False


# CREATE CONTROLLER
continue_training = False
controller_type = 'train' # debug train evaluate empty
controller = utils.get_controller(
	controller_type = controller_type,
	total_timesteps = 1_000_000, # optional if using train - all other hypers set from model instance
	continue_training = continue_training, # if True will continue learning loop from last step saved, if False will reset learning loop
	model_component = 'Model', # if using train, set model
	environment_component = 'TrainEnvironment', # if using train, set train environment
	evaluator_component = 'Evaluator', # if using train (optional) or evaluate, set evaluator component
	tb_log_name = 'run', # logs tensor board to this directory
	)
# read old config file?
read_config = continue_training
read_configuration_path = utils.get_global_parameter('working_directory') + 'configuration.json'
# read old RL model?
read_model = continue_training
read_model_path = utils.get_global_parameter('working_directory') + 'model.zip'
# read old replay buffer data?
read_replay_buffer = continue_training
read_replay_buffer_path = utils.get_global_parameter('working_directory') + 'replay_buffer.pkl'


# READ OLD CONFIGURATION FILE
if read_config:
	# load configuration file and create object to save and connect components
	configuration = Configuration.load(read_configuration_path)
	if update_meta:
		configuration.update_meta(meta)
	Configuration.set_active(configuration)
	configuration.set_controller(controller)


# MAKE NEW CONFIGURATION
elif not read_config:
	# create new configuration object to save and connect components
	configuration = Configuration(meta)
	Configuration.set_active(configuration)
	configuration.set_controller(controller)
	

	# **** SET PARAMETERS ****
	# RL model to use
	model = 'Hyper' # DQN A2C DDPG PPO SAC TD3 Hyper
	# set drone type to use
	drone = 'AirSim' # AirSim Tello
	# set sensors to use
	image_sensors = [
		'Camera', 
		]
	# image shape is set here
	image_bands = 1
	image_height = 84 
	image_width = 84 
	vector_sensors = [
		#'Distance', # [1]
		#'DronePosition', # [3]
		#'DroneOrientation', # [1]
		#'GoalPosition', # [3]
		#'GoalOrientation', # [1]
		#'DroneToGoalPosition', # [3]
		'DroneToGoalDistance', # [1]
		'DroneToGoalOrientation', # [1]
		'FlattenedCamera', # [x]
		]
	flattened_camera_length = 5
	# vector shape is hard coded
	vector_length = 7
	# set number of timesteps to keep in current state
	nTimesteps = 4
	# set modality being used
	observation = 'Vector' # Image Vector Multi
	# set observer component to handle the observation space
	observer = 'Multi' if observation == 'Multi' else 'Single'
	# detrmine to include z-axis (vertical) in objective during calulations
	include_z = False
	# voxels to check valid spawn/objective points on map and visualize results (optional)
	use_voxels = True
	# set max steps
	max_steps = 14
	# set tolerance to reach goal within (arbitrary units depending on drone)
	goal_tolerance = 4
	# set action space type
	action_type = 'continuous' # discrete continuous
	# how many episodes in each evaluation set?
	num_eval_episodes = 6
	# how many training episode before we evaluate/update?
	evaluate_frequency = 100
	# bounds on map (where the drone can go)
	x_bounds = [-100, 100]
	y_bounds = [-100, 100]
	z_bounds = [-100, 100]
	

	# **** CREATE COMPONENTS ****

	# GOAL - make sure to add this to your environment others_components
	from datastructs.relativegoal import RelativeGoal
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
		z_bounds = [-4, -4],
		random_yaw_on_train = False,
		random_yaw_on_evaluate = False,
		random_yaw_min = -1 * math.pi,
		random_yaw_max = math.pi,
		reset_on_step = False,
		name = 'Goal',
		)

	# MAP - controls the map that the drone agent will be traversing
	if drone == 'AirSim':
		from maps.airsimmap import AirSimMap
		AirSimMap(
			voxels_component='Voxels' if use_voxels else None,
			settings = {
				'LocalHostIp': '127.0.0.1',
				'ApiServerPort': 41451,
				'ClockSpeed': 16,
				#"ViewMode": "NoDisplay",
				},
			settings_directory = 'maps/airsim_settings/',
			setting_files = [
				'lightweight', 
				'tellocamera', 
				#'bellydistance',
				],
			release_directory = 'local/airsim_maps/',
			release_name = 'Blocks',
			console_flags = [
				'-Windowed',
				#'-RenderOffscreen',
			],
			name = 'Map',
		)
	# deploying to a field with no connectivity to the program, dummy object
	elif drone == 'Tello':
		from maps.field import Field
		map_ = Field(
			voxels_component='Voxels' if use_voxels else None,
		)

	# VOXELS - 2d representation of map (not required)
	#	usefull in visualizations and checkin spawn/objective points
	if use_voxels:
		from datastructs.voxels import Voxels
		if drone == 'AirSim':
			Voxels(relative_path = (
				utils.get_global_parameter('working_directory')
				+ 'map_voxels.binvox'
				),
					map_component = 'Map',
					make_new = True,
					floor_z = None,
					center = [0,0,0],
					resolution = 1,
					x_length = 200,
					y_length = 200,
					z_length = 100,
					name = 'Voxels',
				)

	# DRONE
	if drone == 'AirSim':
		from drones.airsimdrone import AirSimDrone
		AirSimDrone(
			name='Drone',
		)
	elif drone == 'Tello':
		from drones.tello import Tello 
		Tello(
			wifi_name = 'cloud',
			wifi_password = 'bustersword',
			name='Drone',
		)

	# TRANSFORMER, 0 reserved for bad data
	from transformers.gaussiannoise import GaussianNoise
	GaussianNoise(
		mean = 0,
		deviation = 0.5,
		name = 'PositionNoise',
	)
	GaussianNoise(
		mean = 0,
		deviation = math.radians(5),
		name = 'OrientationNoise',
	)
	from transformers.gaussianblur import GaussianBlur
	GaussianBlur(
		sigma = 4,
		name = 'DepthNoise',
	)
	from transformers.normalize import Normalize
	Normalize(
		min_input = -100, # min distance
		max_input = 100, # max distance
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
		max_input = 100, # max depth
		min_output = 1, # SB3 uses 0-255 pixel values
		max_output = 255, # SB3 uses 0-255 pixel values
		name = 'NormalizeDepth',
	)
	from transformers.resizeimage import ResizeImage
	ResizeImage(
		image_shape = (image_height, image_width),
		name = 'ResizeImage',
	)
	from transformers.resizeflat import ResizeFlat
	ResizeFlat(
		length = flattened_camera_length,
		max_row = 42,
		name = 'ResizeFlat',
	)

	# SENSOR
	if drone == 'AirSim' and 'Camera' in image_sensors:
		from sensors.airsimcamera import AirSimCamera
		AirSimCamera(
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
			prefix = 'drone_to_goal',
			transformers_components = [
				'PositionNoise',
				'NormalizePosition',
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
	if 'FlattenedCamera' in vector_sensors:
		from sensors.airsimcamera import AirSimCamera
		AirSimCamera(
			camera_view = '0',
			image_type = 2,
			as_float = True,
			compress = False,
			is_gray = True,
			transformers_components = [
				'ResizeFlat',
				'PositionNoise',
				'NormalizePosition',
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

	# ACTION
	if action_type == 'discrete':
		move_speed = 2 
		yaw_rate = 11.25
		step_duration = 2 
		from actions.fixedmove import FixedMove 
		FixedMove(
			drone_component = 'Drone', 
			x_speed = move_speed, 
			duration = step_duration,
			name = 'MoveForward',
		)
		from actions.fixedrotate import FixedRotate 
		FixedRotate(
			drone_component = 'Drone',  
			yaw_rate = yaw_rate,
			duration = step_duration,
			name = 'RotateRight',
		)
		FixedRotate(
			drone_component = 'Drone',  
			yaw_rate = -1 * yaw_rate,
			duration = step_duration,
			name = 'RotateLeft',
		)
		FixedRotate(
			drone_component = 'Drone',  
			yaw_rate = yaw_rate * 2,
			duration = step_duration,
			name = 'RotateRight2',
		)
		FixedRotate(
			drone_component = 'Drone',  
			yaw_rate = -1 * yaw_rate * 2,
			duration = step_duration,
			name = 'RotateLeft2',
		)
	elif action_type == 'continuous':
		base_move_speed = 4
		base_yaw_rate = 22.5
		step_duration = 2 
		from actions.move import Move 
		Move(
			drone_component = 'Drone', 
			base_x_speed = base_move_speed, 
			duration = step_duration,
			zero_threshold = 0.25,
			name = 'MoveForward',
		)
		from actions.rotate import Rotate 
		Rotate(
			drone_component = 'Drone',  
			base_yaw_rate = base_yaw_rate,
			duration = step_duration,
			zero_threshold = 0.25,
			name = 'Rotate',
		)

	# ACTOR
	if action_type == 'discrete':
		from actors.discreteactor import DiscreteActor
		DiscreteActor(
			actions_components=[
				'MoveForward',
				'RotateRight',
				'RotateLeft',
				'RotateRight2',
				'RotateLeft2',
				],
			name='Actor',
		)
	elif action_type == 'continuous':
		from actors.continuousactor import ContinuousActor
		ContinuousActor(
			actions_components=[
				'MoveForward',
				'Rotate',
				],
			name='Actor',
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
			'BoundsReward',
		],
		reward_weights = [
			1,
			1,
			1,
			1,
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
	if observation == 'Image': 
		policy = 'CnnPolicy'
	elif observation == 'Vector': 
		policy = 'MlpPolicy'
	elif observation == 'Multi': 
		policy = 'MultiInputPolicy'
	print('policy', policy)
	policy_kwargs = None
	if model == 'Hyper':
		from models.hyper import Hyper
		Hyper(
			environment_component = 'TrainEnvironment',
			_space = {
				'learning_rate':hp.quniform('learning_rate', 1, 6, 1),
				'tau':hp.uniform('tau', 0, 1.0),
				'gamma':hp.quniform('gamma', 1, 6, 1),
			},
			model_type = 'TD3',
			default_params={
				'policy': policy,
				'policy_kwargs': policy_kwargs,
				'verbose': 0,
			},
			resets_components = [
				'TrainEnvironment',
				'EvaluateEnvironment',
				'Evaluator',
				'StepsReward',
				'StepsTerminator',
			],
			max_evals = 16,
			name='Model',
		)
	elif model == 'DQN':
		from models.dqn import DQN
		DQN(
			environment_component = 'TrainEnvironment',
			policy = policy,
			learning_rate = 1e-4,
			buffer_size = evaluate_frequency * 1000,
			learning_starts = evaluate_frequency,
			batch_size = 32,
			tau = 1.0,
			gamma = 0.99,
			train_freq = 4,
			gradient_steps = 1,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			target_update_interval = evaluate_frequency * 10,
			exploration_fraction = 0.1,
			exploration_initial_eps = 1.0,
			exploration_final_eps = 0.1,
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
			learning_rate = 1e-3,
			buffer_size = evaluate_frequency * 1000,
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
	elif model == 'DDPG':
		from models.ddpg import DDPG
		DDPG(
			environment_component = 'TrainEnvironment',
			policy = policy,
			learning_rate = 1e-3,
			buffer_size = evaluate_frequency * 1000,
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
	'''
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
		save_benchmarks=False,
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
		save_benchmarks=False,
		name='EvaluateSaver',
	)

	# ENVIRONMENT
	from environments.dronerl import DroneRL
	DroneRL(
		drone_component='Drone', 
		actor_component='Actor', 
		observer_component='Observer', 
		rewarder_component='Rewarder', 
		terminators_components=[
			'CollisionTerminator',
			'GoalTerminator',
			'StepsTerminator',
			'BoundsTerminator',
			],
		spawner_component='TrainSpawner',
		goal_component='Goal',
		evaluator_component='Evaluator',
		saver_component='TrainSaver',
		is_evaluation_environment=False,
		name = 'TrainEnvironment',
	)
	DroneRL(
		drone_component='Drone', 
		actor_component='Actor', 
		observer_component='Observer', 
		rewarder_component='Rewarder', 
		terminators_components=[
			'CollisionTerminator',
			'GoalTerminator',
			'StepsTerminator',
			'BoundsTerminator',
			],
		spawner_component='EvaluateSpawner',
		goal_component='Goal',
		evaluator_component=None,
		saver_component='EvaluateSaver',
		is_evaluation_environment=True,
		name = 'EvaluateEnvironment',
	)
utils.speak('configuration created!')


t1 = time()
# CONNECT COMPONENTS
configuration.connect_all()
if model == 'DQN':
	print(configuration.get_component('Model')._sb3model.q_net)
if model == 'DDPG' or model == 'TD3':
	print(configuration.get_component('Model')._sb3model.critic)

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()
t2 = time()
delta_t = (t2 - t1) / 3600
print('ran in', delta_t, 'hours')


# done
configuration.controller.stop()
