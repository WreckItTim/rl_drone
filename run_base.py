import utils
from configuration import Configuration
import math
repo_version = 'gamma8'


# runs some overarching base things
def create_base_components(
		run_name, # string value for output 
		continue_training=False, # set to true if continuing training from checkpoint
		controller_type='Debug', # Train, Debug, Drift, Evaluate
		airsim_release = 'Blocks', # name of airsim release to use, see maps.arisimmap
		include_z=True, # includes z-axis in calculations (such as distance to goal)
		clock_speed=1, # speed to run simulation (warning: speed correlates to sim quality)
		flat=None, # flatten depth map to input into MLP
			# if flat is not None then it should be a 2-d array such as:
			# [[16, 32, 52, 68, 84], [21, 42, 63, 84]]
			# where first array is flat_rows and second is flat_cols
			# see transformers.resizeflat
):

	# **** SETUP ****
	clock_speed = 8

	# get OS, set file IO paths
	OS = utils.setup(
		write_parent = 'local/runs/',
		run_prefix = repo_version + '_' + run_name,
		)
	working_directory = utils.get_global_parameter('working_directory')

	# CREATE CONTROLLER
	controller = utils.get_controller(
		controller_type = controller_type,
		total_timesteps = 1_000_000, # optional if using train - all other hypers set from model instance
		continue_training = continue_training, # if True will continue learning loop from last step saved, if False will reset learning loop
		model_component = 'Model', # if using train, set model
		environment_component = 'TrainEnvironment', # if using train, set train environment
		tb_log_name = 'tb_log', # logs tensor board to this directory
		)

	# SET META DATA (anything you want here, just writes to config file as a dict)
	meta = {
		'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
		'repo_version': repo_version,
		'run_name': utils.get_global_parameter('run_name'),
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
			if airsim_release == 'Blocks':
				release_path = 'local/airsim_maps/Blocks/WindowsNoEditor/Blocks.exe'
			if airsim_release == 'AirSimNH':
				release_path = 'local/airsim_maps/AirSimNH/WindowsNoEditor/AirSimNH.sh'
		if utils.get_global_parameter('OS') == 'linux':
			if airsim_release == 'Blocks':
				release_path = 'local/airsim_maps/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
			if airsim_release == 'AirSimNH':
				release_path = 'local/airsim_maps/AirSimNH/LinuxNoEditor/AirSimNH.sh'
		console_flags = []
		render_screen = utils.get_global_parameter('render_screen')
		if render_screen:
			console_flags.append('-Windowed')
		else:
			console_flags.append('-RenderOffscreen')
		AirSimMap(
			voxels_component='Voxels',
			release_path = release_path,
			settings = {
				'ClockSpeed': clock_speed,
				},
			setting_files = [
				'lightweight', 
				],
			console_flags = console_flags.copy(),
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
			name='Drone',
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
			include_z = include_z,
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

		# CREATE OBSERVATION SPACE
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
		# images are 256 x 144 (width x height)
		AirSimCamera(
			airsim_component = 'Map',
			transformers_components = [
				'ResizeImage',
				'DepthNoise',
				'NormalizeDepth',
				],
			name = 'DepthMap',
			)
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
			min_input = 0, # min depth
			max_input = max_distance, # max depth
			min_output = 1, # SB3 uses 0-255 pixel values
			max_output = 255, # SB3 uses 0-255 pixel values
			name = 'NormalizeDepth',
		)
		Normalize(
			max_input = 2*math.pi, # max angle
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
			max_cols = flat[0] if flat is not None else [],
			max_rows = flat[1] if flat is not None else [],
			name = 'ResizeFlat',
		)

		# ACTIONS
		base_move_distance = 4 # meters, will multiple rl_output by this value
		min_move_distance = 1 # meters, will not issue a command below this value (true zero)
		base_yaw_deg = 90 # degrees, will multiple rl_output by this value
		min_yaw_deg = 10 # degrees, will not issue a command below this value (true zero)
		speed = 2 # meters/second (2 is a brisk walking speed)
		from actions.move import Move 
		Move(
			drone_component = 'Drone', 
			base_x_rel = base_move_distance, 
			speed = speed,
			zero_threshold = min_move_distance/base_move_distance,
			name = 'MoveForward',
		)
		Move(
			drone_component = 'Drone', 
			base_z_rel = base_move_distance, 
			speed = speed,
			zero_threshold = min_move_distance/base_move_distance,
			min_space = -1, # allows up and down movements
			name = 'MoveVertical',
		)
		from actions.rotate import Rotate 
		Rotate(
			drone_component = 'Drone',  
			base_yaw_deg = base_yaw_deg,
			zero_threshold = min_yaw_deg/base_yaw_deg,
			name = 'Rotate',
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
		# TRACKER - tracks resources on local computer
		from modifiers.tracker import Tracker
		Tracker(
			base_component = 'TrainEnvironment',
			parent_method = 'reset',
			track_vars = [
						'gpu', 
						'ram',
						'cpu',
						'proc',
						],
			order = 'post',
			save_every = checkpoint,
			frequency = 1,
			name='Tracker',
		)
	return configuration