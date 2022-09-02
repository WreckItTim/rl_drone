import os
from utils import *
from component import *
import random

# USER PARAMETERS and setup'
repo_version = 'gamma'
test_type =  'debug' # 'debug' 'alpha' 'beta' 'gamma'
model_type = 'DQN' # A2C DDPG DQN PPO SAC TD3 (make sure you are using the correct action and observer types for the given model)
train_or_evaluate = 'train'
run_name = test_type + '_' + model_type + '_' + train_or_evaluate
timestamp = get_timestamp() # timestamp used for default write folder, also used to stamp configuration file if write_configuration=True
write_folder = f'temp/' + run_name + '/' # f'temp/{timestamp}/' # will write any files to this folder
read_configuration = train_or_evaluate == 'evaluate' # True: read configuration file to create components
read_configuration_path = 'temp/' + test_type + '_' + model_type + '_train/configuration.json' # path to read configuration file if read_configuration=True
write_configuration = True # True: writes new configuration file after creating all components
write_configuration_path = write_folder + '/configuration.json' # path to write a new configuration file if write_configuration=True
read_global_parameters() # True: sets all variables in the global_parameters.json file
if not os.path.exists('temp/'):
	os.makedirs('temp/')
if not os.path.exists(write_folder):
	os.makedirs(write_folder)
global_parameters['write_folder'] = write_folder # all components will write files/sub_directories to this master folder, unless you otherwise specify an absolute path


# READ OLD CONFIGURATION FILE, you need do nothing else if path is set correctly above
if read_configuration==True:
	speak('reading configuration...')
	configuration = read_json(read_configuration_path)
	controller, components, timestamp, repo_version = deserialize_configuration(configuration) # note that this timestamp is from when the configuration file was last updated (changed when written)

	# ALTER ANY READ-IN COMPONENTS or make a new controller as needed here
	from controllers.evaluaterl import EvaluateRL
	controller = EvaluateRL(
		model_component=model_type + '__1',
		n_eval_episodes=4,
	)


# MAKE NEW CONFIGURATION, set parameters, and create component objects one by one by code, as needed below (all packaged components are listed below with __init__ args)
if read_configuration==False:
	speak('creating new configuration...')
	
	# global parameters to be used by all components
	controller = 'Debug' # Debug TrainRL EvaluateRL
	drone = 'AirSim' # AirSim Tello
	model = 'DQN' # DQN A2C DDPG PPO SAC TD3
	observer = 'MultiStack' # Single SingleLag MultiStack
	sensors = [
		'Camera', 
		'Distance',
		]
	output_height = 84 # output shape after processing
	output_width = 84 # output shape after processing
	relative_objective_point = (100, 0, 0)
	start_z = -5
	every_nEpisodes = 100
	step_size = 4 # meters
	speed = 4 # meters / second
	from datastructs.zone import Zone
	spawn_zones = [
		Zone(
			x_min=-10, 
			x_max=10, 
			y_min=-10, 
			y_max=10, 
			z_min=start_z, 
			z_max=start_z,
			name='SpawnZone',
			),
	]

	# MAP
	if drone == 'AirSim':
		from maps.airsimmap import AirSimMap
		AirSimMap(
			settings = None,
			settings_directory = 'maps/airsim_settings/',
			setting_files = [
				'lightweight', 
				'speedup', 
				'tellocamera', 
				'bellydistance'
				],
			release_directory = 'resources/airsim_maps/',
			release_relative_path = 'Blocks/',
			release_name = 'Blocks.exe',
			name = 'Map',
		)
	elif drone == 'Tello':
		from maps.field import Field
		map_ = Field(
			name='Map',
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

	# SENSOR
	if drone == 'AirSim' and 'Camera' in sensors:
		from sensors.airsimcamera import AirSimCamera
		AirSimCamera(
			camera_view = '0',
			image_type = 2,
			as_float = True,
			compress = False,
			is_gray = True,
			name = 'Camera',
		)
	if drone == 'AirSim' and 'Distance' in sensors:
		from sensors.airsimdistance import AirSimDistance
		AirSimDistance(
			name = 'Distance',
		)
	if drone == 'Tello' and 'Camera' in sensors:
		from sensors.portcamera import PortCamera
		PortCamera(
			port = 'udp://0.0.0.0:11111',
			is_gray = False,
			name = 'Camera',
		)

	# TRANSFORMER
	from transformers.resizeimage import ResizeImage
	ResizeImage(
		image_shape = (output_height, output_width, 1),
		name = 'ResizeImage',
	)
	from transformers.normalizedepth import NormalizeDepth
	NormalizeDepth(
		min_depth = 0,
		max_depth = 100,
		name = 'NormalizeDepth',
	)

	# OBSERVER
	if observer == 'Single':
		from observers.single import Single
		Single(
			sensor_component = sensors[0], 
			transformers_components = [
				'ResizeImage', 
				'NormalizeDepth',
				],
			output_height = output_height,
			output_width = output_width,
			name = 'Observer',
		)
	elif observer == 'SingleLag':
		from observers.singlelag import SingleLag
		SingleLag(
			sensor_component = sensors[0], 
			transformers_components = [
				'ResizeImage', 
				'NormalizeDepth',
				],
			n_frames_lag = 2,
			output_height = output_height,
			output_width = output_width,
			name = 'Observer',
		)
	elif observer == 'MultiStack':
		distance_thickness = 20
		resize = ResizeImage(
			image_shape=(output_height-distance_thickness, output_width, 1),
		)
		normalize_distance = NormalizeDepth(
			min_depth = 0,
			max_depth = 40,
		)
		from observers.single import Single
		observers = [
			Single(
				sensor_component = 'Camera', 
				transformers_components = [
					resize,
					'NormalizeDepth',
				],
				output_height = output_height-distance_thickness,
				output_width = output_width,
			),
			Single(
				sensor_component = 'Distance', 
				transformers_components = [
					normalize_distance,
				],
				output_height = distance_thickness,
				output_width = output_width,
			),
			]
		from observers.multistack import MultiStack
		MultiStack(
			observers_components = observers,
			output_height = output_height,
			output_width = output_width,
			stack='v',
			name='Observer',
		)

	# ACTION
	from actions.fixedmove import FixedMove 
	FixedMove.get_move(
		drone_component = 'Drone', 
		move_type = 'Up', 
		step_size = step_size,
		speed = speed,
	)
	FixedMove.get_move(
		drone_component = 'Drone', 
		move_type = 'Down', 
		step_size = step_size,
		speed = speed,
	)
	FixedMove.get_move(
		drone_component = 'Drone', 
		move_type = 'Forward', 
		step_size = step_size,
		speed = speed,
	)

	# ACTOR
	from actors.discreteactor import DiscreteActor
	DiscreteActor(
		actions_components=['Up','Down','Forward',],
		name='Actor',
	)

	# REWARD
	from rewards.avoid import Avoid
	Avoid(
		drone_component = 'Drone',
		name = 'AvoidReward',
	)
	from rewards.relativepoint import RelativePoint
	RelativePoint(
		drone_component = 'Drone',
		xyz_point = relative_objective_point,
		min_distance = 5,
		max_distance = 200,
		name = 'RelativePointReward',
	)

	# REWARDER
	from rewarders.schema import Schema
	Schema(
		rewards_components = [
			'AvoidReward',
			'RelativePointReward',
			],
		reward_weights = [
			1,
			2,
			],
		name = 'Rewarder',
	)

	# TERMINATOR
	from terminators.collision import Collision
	Collision(
		drone_component = 'Drone',
		name = 'Collision',
	)
	from terminators.relativepoint import RelativePoint
	RelativePoint(
		drone_component = 'Drone',
		xyz_point = relative_objective_point,
		min_distance = 5,
		max_distance = 110,
		name = 'RelativePointTerminator',
	)
	from terminators.rewardthresh import RewardThresh
	RewardThresh(
		min_reward = 0,
		name = 'RewardThresh',
	)
	from terminators.maxsteps import MaxSteps
	MaxSteps(
		max_steps = 64,
		name = 'MaxSteps',
	)

	# MODEL
	if model == 'DQN':
		from models.dqn import DQN
		DQN(
			environment_component = 'Environment',
			policy = 'CnnPolicy',
			learning_rate = 1e-4,
			buffer_size = every_nEpisodes*10,
			learning_starts = every_nEpisodes,
			batch_size = 32,
			tau = 1.0,
			gamma = 0.99,
			train_freq = 4,
			gradient_steps = 1,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			target_update_interval = every_nEpisodes,
			exploration_fraction = 0.1,
			exploration_initial_eps = 1.0,
			exploration_final_eps = 0.05,
			max_grad_norm = 10,
			tensorboard_log = None,
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
	elif model == 'A2C':
		from models.a2c import A2C
		A2C(
			environment_component = 'Environment',
			policy = 'CnnPolicy',
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
			tensorboard_log = None,
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
			environment_component = 'Environment',
			policy = 'CnnPolicy',
			learning_rate = 1e-3,
			buffer_size = 1_000_000,
			learning_starts = 100,
			batch_size = 100,
			tau = 0.005,
			gamma = 0.99,
			train_freq = (1, "episode"),
			gradient_steps = -1,
			action_noise = None,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			tensorboard_log = None,
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
			environment_component = 'Environment',
			policy = 'CnnPolicy',
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
			tensorboard_log = None,
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
			environment_component = 'Environment',
			policy = 'CnnPolicy',
			learning_rate = 1e-3,
			buffer_size = 1_000_000,
			learning_starts = 100,
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
			tensorboard_log = None,
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
			environment_component = 'Environment',
			policy = 'CnnPolicy',
			learning_rate = 1e-3,
			buffer_size = 1_000_000,
			learning_starts = 100,
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
			tensorboard_log = None,
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

	# OTHER
	from others.randomspawnpoint import RandomSpawnPoint
	RandomSpawnPoint(
		drone_component='Drone', 
		environment_component='Environment',
		spawn_zones_components=spawn_zones,
		name='RandomSpawnPoint',
	)
	from others.randomspawnyaw import RandomSpawnYaw
	RandomSpawnYaw(
		drone_component='Drone', 
		environment_component='Environment',
		yaw_min=0, 
		yaw_max=360,
		name='RandomSpawnYaw',
	)
	from others.spawnevaluator import SpawnEvaluator
	SpawnEvaluator(
		model_component='Model',
		drone_component='Drone',
		environment_component='Environment',
		evaluate_every_nEpisodes=every_nEpisodes,
		nTimes=4, 
		spawns=([[0,0,start_z],0], [[0,0,start_z],135], [[0,0,start_z],180], [[0,0,start_z],225]),
		name='SpawnEvaluator',
	)
	from others.modelsaver import ModelSaver
	ModelSaver(
		model_component='Model',
		environment_component='Environment',
		save_every_nEpisodes=every_nEpisodes,
		name='ModelSaver',
	)
	from others.replaybuffersaver import ReplayBufferSaver
	ReplayBufferSaver(
		model_component='Model',
		environment_component='Environment',
		save_every_nEpisodes=every_nEpisodes,
		name='ReplayBufferSaver',
	)
	from others.benchmarker import BenchMarker
	BenchMarker(
		environment_component='Environment',
		benchmark_every_nEpisodes=every_nEpisodes,
		name='BenchMarker',
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
			'RelativePoint',
			#'RewardThresh',
			'MaxSteps',
			],
		others_components=[
			'RandomSpawnPoint',
			'RandomSpawnYaw',
			'SpawnEvaluator',
			'ModelSaver',
			'ReplayBufferSaver',
			'BenchMarker',
			],
		name = 'Environment'
	)

	# CONTROLLER
	if controller == 'Debug':
		from controllers.debug import Debug
		controller = Debug(
		drone_component='Drone'
	)
	elif controller == 'TrainRL':
		from controllers.trainrl import TrainRL
		controller = TrainRL(
		model_component='Model',
	)
	elif controller == 'EvaluateRL':
		from controllers.evaluaterl import EvaluateRL
		controller = EvaluateRL(
		model_component='Model',
		n_eval_episodes=3,
	)

speak('configuration loaded!')


# FETCH COMPONENTS (before any are created during run)
configuration_components = get_all_components()


# WRITE CONFIGURATION
if write_configuration:
	configuration = serialize_configuration(controller, configuration_components, timestamp, repo_version)
	write_json(configuration, write_configuration_path)
	write_json(configuration, 'configurations/last.json')
 

# CONNECT COMPONENTS
connect_components(configuration_components)
speak('components connected!')


# CONNECT CONTROLLER
controller.connect()
speak('controller connected!')


# RUN CONTROLLER
speak('running controller...')
controller.run()


# FETCH COMPONENTS (including any created during run)
run_components = get_all_components()


# LOG BENCHMARKS
benchmark_components(run_components)
speak('components benchmarked!')


# DISCONNECT COMPONENTS
disconnect_components(run_components)
speak('components disconnected!')


# DISCONNECT CONTROLLER
controller.disconnect()
speak('controller disconnected!')


# ALL DONE
speak('Good bye!')