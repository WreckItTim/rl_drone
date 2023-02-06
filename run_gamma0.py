import utils
from configuration import Configuration
import math


# **** SETUP ****

# get OS, set file IO paths
run_name = 'gamma4_gamma0_hackfest4_run1' # subcategory of test type
OS = utils.setup(
	working_directory = 'local/runs/' + run_name + '/',
	)
working_directory = utils.get_global_parameter('working_directory')

# CREATE CONTROLLER
continue_training = False
controller = utils.get_controller(
	controller_type = 'train',
	total_timesteps = 1_000_000, # optional if using train - all other hypers set from model instance
	continue_training = continue_training, # if True will continue learning loop from last step saved, if False will reset learning loop
	model_component = 'Model', # if using train, set model
	environment_component = 'TrainEnvironment', # if using train, set train environment
	tb_log_name = 'tb_log', # logs tensor board to this directory
	)

# SET META DATA (anything you want here, just writes to config file as a dict)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'repo_version': 'gamma4',
	'run_name': run_name,
	'timestamp': utils.get_timestamp(),
	'run_OS': utils.get_global_parameter('OS'),
	'absolute_path' : utils.get_global_parameter('absolute_path'),
	'working_directory' : working_directory,
	}


# READ CONFIGURATION
read_configuration_path = working_directory + 'configuration.json'
update_meta = True
if continue_training:
	# load configuration file and create object to save and connect components
	configuration = Configuration.load(read_configuration_path, controller)
	if update_meta:
		configuration.update_meta(meta)
	# load model weights and replay buffer
	read_model_path = working_directory + 'Model/model.zip'
	read_replay_buffer_path = working_directory + 'Model/replay_buffer.zip'
	_model = configuration.get_component('Model')
	_model.read_model_path = read_model_path
	_model.read_replay_buffer_path = read_replay_buffer_path


# or CREATE CONFIGURATION
else:
	# will add components to this configuration automatically
	# can switch which configuration is active to add to different once
	# I almost always just use 1 configuration per run
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
		#release_path = 'local/airsim_maps/AirSimNH/WindowsNoEditor/AirSimNH.sh'
	if utils.get_global_parameter('OS') == 'linux':
		release_path = 'local/airsim_maps/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
		#release_path = 'local/airsim_maps/AirSimNH/LinuxNoEditor/AirSimNH.sh'
	AirSimMap(
		voxels_component='Voxels',
		release_path = release_path,
		settings = {
			'ClockSpeed': 4,
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
		relative_path = working_directory + 'map_voxels.binvox',
		map_component = 'Map',
		name = 'Voxels',
		)


	# CREATE DRONE
	from drones.airsimdrone import AirSimDrone
	AirSimDrone(
		airsim_component = 'Map',
		drift_stop_gap = False,
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
		tensorboard_log = working_directory + 'tensorboard/',
		overide_memory = True, # memory benchmark on
		name='Model',
	)


	# CREATE GOAL
	max_distance = 100
	x_bounds = [-1*max_distance, max_distance]
	y_bounds = [-1*max_distance, max_distance]
	z_bounds = [-4, -4]
	from others.relativegoal import RelativeGoal
	RelativeGoal(
		drone_component = 'Drone',
		map_component = 'Map',
		xyz_point = [6, 6, 0],
		random_point_on_train = True,
		random_point_on_evaluate = False,
		random_dim_min = 4,
		random_dim_max = 8,
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
		sensors_components = [
			'GoalDistance', 
			'GoalOrientation', 
			'FlattenedDepth', 
			'Moves',
			], 
		vector_length = 1 + 1 + len(flat_cols) + len(actions),
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
		include_z = False,
		name = 'GoalReward',
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
	# ALTITUDE ADJUSTER (for horizontal motion, 
		# since moving forward adds drift up)
	from modifiers.altadjust import AltAdjust
	AltAdjust(
		base_component = 'Actor',
		parent_method = 'step',
		drone_component = 'Drone',
		order = 'post',
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
	from modifiers.tracker import Tracker
	Tracker(
		base_component = 'TrainEnvironment',
		parent_method = 'end',
		track_vars = [
					  'gpu', 
					  'ram',
					  'cpu',
					  ],
		order = 'pre',
		write_path = working_directory + 'track_log.json',
		frequency = 10,
		activate_on_first = True,
		name='Tracker',
	)


utils.speak('configuration created!')


stopwatch = utils.StopWatch()
# CONNECT COMPONENTS
configuration.connect_all()
model_name = str(configuration.get_component('Model')._child())
sb3_model = configuration.get_component('Model')._sb3model
if 'dqn' in model_name:
	print(sb3_model.q_net)
	for name, param in sb3_model.q_net.named_parameters():
		print(name, param)
if 'ddpg' in model_name or 'td3' in model_name:
	print(sb3_model.critic)
	for name, param in sb3_model.critic.named_parameters():
		print(name, param)
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
