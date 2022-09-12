import os
import utils
from configuration import Configuration
import math
import numpy as np


# USER PARAMETERS and SETUP
# test version is just a name used for logging (optional)
test_version =  'temp5'
# select name of reinforcement learning model to use
model = 'DQN' # DQN A2C DDPG PPO SAC TD3
# set the controller type to use
controller_type = 'train' # train evaluate debug
# create run name (not unique) for logging (optional)
run_name = test_version + '_' + model + '_' + controller_type
# create working directory to read/write files to
working_directory = f'temp/' + run_name + '/'
# path to read configuration file from, if desired (optional)
read_configuration_path = 'temp/' + test_version + '_' + model + '_train/configuration.json'
# tell program to make a new configuration, if False will read an old one from read_configuration_path
make_new_configuration = True

# make temp folder if not exists - required
if not os.path.exists('temp/'):
	os.makedirs('temp/')
# make working directory if not exists
if not os.path.exists(working_directory):
	os.makedirs(working_directory)
# save working directory path to global_parameters to be visible by all 
utils.set_global_parameter('working_directory', working_directory) # relative to repo
# absoulte path on local computer to repo
utils.set_global_parameter('absolute_path',  os.getcwd() + '/') # end all folder paths with /


# META data to log in configuration file - no required format, anything you want to note here
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'timestamp': utils.get_timestamp(),
	'repo_version': 'eta3',
	'run_name': run_name
	}
# select rather to overwrite meta data in configuration file (only if reading one)
update_meta = False

# learning params
total_timesteps = 1_000_000
every_nEpisodes = 400

# create CONTROLLER - controls all components (mode)
# debug mode will prompt user input for which component(s) to debug
if controller_type == 'debug':
	from controllers.debug import Debug
	controller = Debug(
		drone_component='Drone',
		)
# train will create a new or read in a previously trained model
# set continue_training=True to pick up where learning loop last saved
# or set continue_training=False to keep weights but start new learning loop
elif controller_type == 'train':
	from controllers.trainrl import TrainRL
	controller = TrainRL(
		model_component = 'Model',
		environment_component = 'TrainEnvironment',
		evaluator_component = 'Evaluator',
		total_timesteps = total_timesteps,
		callback = None,
		log_interval = -1,
		tb_log_name = 'phase1',
		eval_env = None,
		eval_freq = -1,
		n_eval_episodes = -1,
		eval_log_path = None,
		continue_training = True,
		)
# evaluate willl read in a trained model and evaluate on given environment
elif controller_type == 'evaluate':
	from controllers.evaluaterl import EvaluateRL
	controller = EvaluateRL(
		model_component= 'Model',
		)


# READ OLD CONFIGURATION FILE
# you do not need to do nothing else if reading a config file as is
if not make_new_configuration and os.path.exists(read_configuration_path):
	# load configuration file and create object to save and connect components
	configuration = Configuration.load(read_configuration_path)
	if update_meta:
		configuration.update_meta(meta)
	Configuration.set_active(configuration)
	configuration.set_controller(controller)


# MAKE NEW CONFIGURATION
# all packaged componets are listed here and created for debugging purposes
# the below configuration is set to run the Delta Demonstration in our paper
# edit as needed, suggested to use as a template
else:
	# create new configuration object to save and connect components
	configuration = Configuration(meta)
	Configuration.set_active(configuration)
	configuration.set_controller(controller)
	
	# **** SET PARAMETERS ****
	# set drone type to use
	drone = 'AirSim' # AirSim Tello
	# set sensors to use
	image_sensors = [
		'Camera', 
		]
	# image shape is hard coded
	image_bands = 1
	image_height = 84 
	image_width = 84 
	vector_sensors = [
		#'Distance',
		'DronePosition',
		'DroneOrientation',
		'GoalPosition',
		'GoalOrientation',
		'DroneToGoalDistance',
		'DroneToGoalYaw',
		]
	# vector shape is hard coded
	vector_length = 13
	# set number of timesteps to keep in current state
	nTimesteps = 4
	# set modality being used
	observation = 'Multi' # Image Vector Multi
	# set observer component to handle the observation space
	observer = 'Multi' if observation == 'Multi' else 'Single'
	# detrmine to include z-axis (vertical) in objective during calulations
	include_z = False
	# set starting height of drone for each episode
	start_z = -4 
	# set drone speed for steps in meters / second
	move_speed = 2 
	# set rotate speed for steps in degrees / second
	yaw_rate = 11.25
	# set drone duration of each step in seconds
	step_duration = 2 
	# control if load voxels in to check valid spawn/objective points and visualize results
	use_voxels = True
	# set goal (objective point) - can be relative or absolute
	goal = [8, 0, 0]
	goal_tolerance = 2
	max_steps = 16
	max_distance = math.sqrt(2)*np.linalg.norm(goal)
	

	# **** CREATE COMPONENTS ****

	# GOAL - make sure to add this to your environment others_components
	from datastructs.relativegoal import RelativeGoal
	RelativeGoal(
		drone_component = 'Drone',
		map_component = 'Map',
		xyz_point = goal,
		random_yaw = True,
		random_yaw_min = -1 * math.pi,
		random_yaw_max = math.pi,
		reset_on_step=False,
		name = 'Goal',
		)

	# MAP - controls the map that the drone agent will be traversing
	if drone == 'AirSim':
		from maps.airsimmap import AirSimMap
		AirSimMap(
			voxels_component='Voxels' if use_voxels else None,
			settings = None,
			settings_directory = 'maps/airsim_settings/',
			setting_files = [
				'lightweight', 
				'speedup', 
				'tellocamera', 
				#'bellydistance',
				#'nodisplay',
				],
			release_directory = 'resources/airsim_maps/Blocks/',
			release_name = 'Blocks.exe',
			name = 'Map',
		)
	# deploying to a field with no connectivity to the program, dummy object
	elif drone == 'Tello':
		from maps.field import Field
		map_ = Field(
			voxels_component='Voxels' if use_voxels else None,
		)

	# VOXELS - 2d representation of map (not required)
	if use_voxels:
		from datastructs.voxels import Voxels
		if drone == 'AirSim':
			Voxels(absolute_path = (
				utils.get_global_parameter('absolute_path') 
				+ utils.get_global_parameter('working_directory')
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

	# TRANSFORMER
	from transformers.normalize import Normalize
	Normalize(
		min_input = -200, # min distance
		max_input = 200, # max distance
		min_output = 0, # SB3 uses 0-1 floating point values
		max_output = 1, # SB3 uses 0-1 floating point values
		name = 'NormalizePosition',
	)
	Normalize(
		min_input = -1 * math.pi, # min angle
		max_input = math.pi, # max angle
		min_output = 0, # SB3 uses 0-1 floating point values
		max_output = 1, # SB3 uses 0-1 floating point values
		name = 'NormalizeOrientation',
	)
	Normalize(
		min_input = 0, # min depth
		max_input = 100, # max depth
		min_output = 0, # SB3 uses 0-255 pixel values
		max_output = 255, # SB3 uses 0-255 pixel values
		name = 'NormalizeDepth',
	)
	from transformers.resizeimage import ResizeImage
	ResizeImage(
		image_shape = (image_height, image_width, image_bands),
		name = 'ResizeImage',
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
				'NormalizeDepth',
				],
			name = 'Camera',
			)
	if drone == 'AirSim' and 'Distance' in vector_sensors:
		from sensors.airsimdistance import AirSimDistance
		AirSimDistance(
			transformers_components = [
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
				'NormalizePosition',
				], 
			name = 'DroneToGoalPosition',
		)
	if 'DroneToGoalOrientation' in vector_sensors:
		from sensors.orientation import Orientation
		Orientation(
			misc_component = 'Drone',
			misc2_component = 'Goal',
			prefix = 'drone_to_goal',
			transformers_components = [
				'NormalizeOrientation',
				],
			name = 'DroneToGoalOrientation',
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

	# ACTOR
	from actors.discreteactor import DiscreteActor
	DiscreteActor(
		actions_components=[
			'MoveForward',
			'RotateRight',
			'RotateLeft',
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
		min_distance = goal_tolerance, 
		max_distance = max_distance,
        goal_tolerance = goal_tolerance,
		include_z = include_z,
		name = 'GoalReward',
	)

	# REWARDER
	from rewarders.schema import Schema
	Schema(
		rewards_components = [
			#'AvoidReward',
			'GoalReward',
			],
		reward_weights = [
			#1,
			1,
			],
		name = 'Rewarder',
	)

	# TERMINATOR
	from terminators.collision import Collision
	Collision(
		drone_component = 'Drone',
		name = 'Collision',
	)
	from terminators.goal import Goal
	Goal(
		drone_component = 'Drone',
		goal_component = 'Goal',
		min_distance = goal_tolerance, 
		max_distance = max_distance,
		include_z = include_z,
		name = 'GoalTerminator',
	)
	from terminators.rewardthresh import RewardThresh
	RewardThresh(
		rewarder_component='Rewarder',
		min_reward = 0,
		name = 'RewardThresh',
	)
	from terminators.maxsteps import MaxSteps
	MaxSteps(
		max_steps = max_steps,
		name = 'MaxSteps',
	)

	# MODEL
	if observation == 'Image': 
		policy = 'CnnPolicy'
	if observation == 'Vector': 
		policy = 'MlpPolicy'
	if observation == 'Multi': 
		policy = 'MultiInputPolicy'
	policy_kwargs = None
	if model == 'DQN':
		from models.dqn import DQN
		DQN(
			environment_component = 'TrainEnvironment',
			policy = policy,
			learning_rate = 1e-4,
			buffer_size = every_nEpisodes * 100,
			learning_starts = every_nEpisodes * 5,
			batch_size = 32,
			tau = 1e-2,
			gamma = 0.9999,
			train_freq = 4,
			gradient_steps = 1,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			target_update_interval = every_nEpisodes,
			exploration_fraction = 0.4,
			exploration_initial_eps = 1.0,
			exploration_final_eps = 0.1,
			max_grad_norm = 10,
			tensorboard_log = working_directory + 'tensorboard/',
			create_eval_env = False,
			policy_kwargs = policy_kwargs,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			write_path = None,
			replay_buffer_path = None,
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
			tensorboard_log = working_directory + 'tensorboard/',
			create_eval_env = False,
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			write_path = None,
			replay_buffer_path = None,
			name='Model',
		)
	elif model == 'DDPG':
		from models.ddpg import DDPG
		DDPG(
			environment_component = 'TrainEnvironment',
			policy = policy,
			learning_rate = 1e-3,
			buffer_size = every_nEpisodes*100,
			learning_starts = every_nEpisodes,
			batch_size = 100,
			tau = 0.005,
			gamma = 0.99,
			train_freq = (1, "episode"),
			gradient_steps = -1,
			action_noise = None,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			tensorboard_log = working_directory + 'tensorboard/',
			create_eval_env = False,
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			write_path = None,
			replay_buffer_path = None,
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
			tensorboard_log = working_directory + 'tensorboard/',
			create_eval_env = False,
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			write_path = None,
			replay_buffer_path = None,
			name='Model',
		)
	elif model == 'SAC':
		from models.sac import SAC
		SAC(
			environment_component = 'TrainEnvironment',
			policy = policy,
			learning_rate = 1e-3,
			buffer_size = every_nEpisodes*100,
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
			tensorboard_log = working_directory + 'tensorboard/',
			create_eval_env = False,
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			write_path = None,
			replay_buffer_path = None,
			name='Model',
		)
	elif model == 'TD3':
		from models.td3 import TD3
		TD3(
			environment_component = 'TrainEnvironment',
			policy = policy,
			learning_rate = 1e-3,
			buffer_size = every_nEpisodes*100,
			learning_starts = every_nEpisodes,
			batch_size = 100,
			tau = 0.005,
			gamma = 0.99,
			train_freq = (1, "episode"),
			gradient_steps = -1,
			action_noise = None,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			policy_delay = 2,
			target_policy_noise = 0.2,
			target_noise_clip = 0.5,
			tensorboard_log = working_directory + 'tensorboard/',
			create_eval_env = False,
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device = "auto",
			init_setup_model = True,
			write_path = None,
			replay_buffer_path = None,
			name='Model',
		)
	

	# SPAWNER
	from others.spawner import Spawner
	from datastructs.spawn import Spawn
	Spawner(
		spawns_components=[
			Spawn(
				map_component = 'Map',
				x_min=-8,
				x_max=8,
				y_min=-8,
				y_max=8,
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
			],
		name='EvaluateSpawner',
	)

	# EVALUATOR
	from others.evaluator import Evaluator
	Evaluator(
		train_environment_component = 'TrainEnvironment',
		evaluate_environment_component = 'EvaluateEnvironment',
		model_component = 'Model',
		frequency = every_nEpisodes,
		nEpisodes = 100,
		stopping_total_success = 4,
		n_success_buffer = 0,
		stopping_improved_steps = 4,
		n_steps_buffer = 4,
		name = 'Evaluator',
	)

	# SAVER
	from others.saver import Saver
	Saver(
		model_component='Model', 
		environment_component='TrainEnvironment',
		frequency=every_nEpisodes, 
		save_model=True,
		save_replay_buffer=True,
		save_configuration_file=True,
		save_benchmarks=True,
		name='Saver',
	)

	# ENVIRONMENT
	from environments.dronerl import DroneRL
	DroneRL(
		drone_component='Drone', 
		actor_component='Actor', 
		observer_component='Observer', 
		rewarder_component='Rewarder', 
		terminators_components=[
			'Collision',
			'GoalTerminator',
			'MaxSteps',
			],
		saver_component='Saver',
		evaluator_component='Evaluator',
		spawner_component='TrainSpawner',
		others_components=[
			'Goal',
			],
		write_observations=False,
		name = 'TrainEnvironment',
	)
	DroneRL(
		drone_component='Drone', 
		actor_component='Actor', 
		observer_component='Observer', 
		rewarder_component='Rewarder', 
		terminators_components=[
			'Collision',
			'GoalTerminator',
			'MaxSteps',
			],
		spawner_component='TrainSpawner',
		others_components=[
			'Goal',
			],
		write_observations=True,
		name = 'EvaluateEnvironment',
	)
utils.speak('configuration created!')


# CONNECT COMPONENTS
configuration.connect_all()
print(configuration.get_component('Model')._sb3model.q_net)
print(configuration.get_component('Model')._sb3model.q_net_target)

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# DISCONNECT COMPONENTS
configuration.disconnect_all()

# all done!
utils.speak('Thatll do pig thatll do')
