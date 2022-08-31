import os
from utils import *
from component import *
import random

# USER PARAMETERS and setup'
repo_version = 'alpha'
test_type = 'beta'
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
	# specify components to be used
	map_type = 'AirSim' # AirSim Field
	drone_type = 'AirSim' # AirSim Tello
	sensor_type = 'AirSimCamera' # AirSimCamera PortCamera
	transformer_types = ['ResizeImage', 'NormalizeDepth'] # ResizeImage NormalizeDepth
	observer_type = 'Single' # Single
	action_type = 'FixedMove' # FixedMove
	move_types = ['up', 'down', 'forward'] # up down forward backward left right (optional for fixed_move, also use any combination-diagnol by placing an '_' to seperate, example: forward_up)
	actor_type = 'DiscreteActor' # DiscreteActor ContinuousActor
	reward_types = ['Avoid', 'RelativePoint'] # Avoid RelativePoint
	rewarder_type = 'Schema' # Schema
	terminator_types = ['Collision', 'RelativePoint', 'MaxSteps'] # Collision RelativePoint RewardThresh MaxSteps
	other_types = ['SpawnEvaluator', 'RandomSpawnPoint', 'RandomSpawnYaw', 'ModelSaver', 'ReplayBufferSaver', 'BenchMarker'] 
	# RandomSpawnPoint RandomSpawnYaw SpawnEvaluator ModelSaver ReplayBufferSaver BenchMarker
	environment_type = 'DroneRL' # DroneRL
	controller_type = 'TrainRL' # TrainRL EvaluateRL Debug (set the debug() method for any component)

	# specify any global parameters to be used by all components
	environment = 'DroneRL' # some components require environment before it is created - so pass in this name and set name during environment init()
	image_shape=(84, 84, 1)
	relative_objective_point=(100, 0, 0)
	start_z = -5
	every_nEpisodes = 1
	from datastructs.zone import Zone
	spawn_zones = [
		Zone(x_min=-10, x_max=10, y_min=-10, y_max=10, z_min=start_z, z_max=start_z),
	]


	# MAP
	if map_type == 'AirSim':
		from maps.airsimmap import AirSimMap
		map_ = AirSimMap(
			settings=None,
			settings_directory='maps/airsim_settings/',
			setting_files=['lightweight', 'speedup', 'cameraresolution', 'nodisplay'],
			release_directory='resources/airsim_maps/',
			release_relative_path='Blocks/',
			release_name='Blocks.exe',
		)
	elif map_type == 'Field':
		from maps.field import Field
		map_ = Field(
		)

	# DRONE
	if drone_type == 'AirSim':
		from drones.airsimdrone import AirSimDrone
		drone = AirSimDrone(
		)
	elif drone_type == 'Tello':
		from drones.tello import Tello 
		drone = Tello(
			wifi_name = 'cloud',
			wifi_password = 'bustersword',
		)

	# SENSOR
	if sensor_type == 'AirSimCamera':
		from sensors.airsimcamera import AirSimCamera
		sensor = AirSimCamera(
			camera_view='0',
			image_type=2,
			as_float=True,
			compress=False,
			is_gray=True
		)
	elif sensor_type == 'PortCamera':
		from sensors.portcamera import PortCamera
		sensor = PortCamera(
			port='udp://0.0.0.0:11111',
			is_gray=False,
		)

	# TRANSFORMER
	transformers_components=[] # robots in disguise! 
	for transformer_type in transformer_types:
		transformer = None
		if transformer_type == 'ResizeImage':
			from transformers.resizeimage import ResizeImage
			transformer = ResizeImage(
				image_shape=image_shape
			)
		elif transformer_type == 'NormalizeDepth':
			from transformers.normalizedepth import NormalizeDepth
			transformer = NormalizeDepth(
				min_depth=0,
				max_depth=100,
			)
		transformers_components.append(transformer)

	# OBSERVER
	if observer_type == 'Single':
		from observers.single import Single
		observer = Single(
			sensor_component=sensor, 
			transformers_components=transformers_components,
			please_write=True, 
			write_directory='temp/',
			output_shape=image_shape,
		)

	# ACTION
	actions_components = []
	if action_type == 'FixedMove':
		from actions.fixedmove import FixedMove 
		for move_type in move_types:
			fixed_move = FixedMove.get_move(
				drone_component=drone, 
				move_type=move_type, 
				step_size=5,
				speed=4,
			)
			actions_components.append(fixed_move)

	# ACTOR
	if actor_type == 'DiscreteActor':
		from actors.discreteactor import DiscreteActor
		actor = DiscreteActor(
			actions_components=actions_components,
		)

	# REWARD
	rewards_components=[]
	for reward_type in reward_types:
		reward = None
		if reward_type == 'Avoid':
			from rewards.avoid import Avoid
			reward = Avoid(
				drone_component = drone
			)
		elif reward_type == 'RelativePoint':
			from rewards.relativepoint import RelativePoint
			reward = RelativePoint(
				drone_component = drone,
				xyz_point = relative_objective_point,
				min_distance = 10,
				max_distance = 190,
			)
		rewards_components.append(reward)

	# REWARDER
	if rewarder_type == 'Schema':
		from rewarders.schema import Schema
		rewarder = Schema(
			rewards_components=rewards_components,
			reward_weights=[1, 2],
		)

	# TERMINATOR
	terminators_components=[]
	for terminator_type in terminator_types:
		terminator = None
		if terminator_type == 'Collision':
			from terminators.collision import Collision
			terminator = Collision(
				drone_component = drone
			)
		elif terminator_type == 'RelativePoint':
			from terminators.relativepoint import RelativePoint
			terminator = RelativePoint(
				drone_component = drone,
				xyz_point = relative_objective_point,
				min_distance = 10,
				max_distance = 110,
			)
		elif terminator_type == 'RewardThresh':
			from terminators.rewardthresh import RewardThresh
			terminator = RewardThresh(
				min_reward = 0,
			)
		elif terminator_type == 'MaxSteps':
			from terminators.maxsteps import MaxSteps
			terminator = MaxSteps(
				max_steps = 50,
			)
		terminators_components.append(terminator)

	# MODEL
	if model_type == 'DQN':
		from models.dqn import DQN
		model = DQN(
			environment_component = environment,
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
		)
	elif model_type == 'A2C':
		from models.a2c import A2C
		model = A2C(
			environment_component = environment,
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
		)
	elif model_type == 'DDPG':
		from models.ddpg import DDPG
		model = DDPG(
			environment_component = environment,
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
		)
	elif model_type == 'PPO':
		from models.ppo import PPO
		model = PPO(
			environment_component = environment,
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
		)
	elif model_type == 'SAC':
		from models.sac import SAC
		model = SAC(
			environment_component = environment,
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
		)
	elif model_type == 'TD3':
		from models.td3 import TD3
		model = TD3(
			environment_component = environment,
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
		)

	# OTHER
	others_components = []
	for other_type in other_types:
		other = None
		if other_type == 'RandomSpawnPoint':
			from others.randomspawnpoint import RandomSpawnPoint
			other = RandomSpawnPoint(
				drone_component=drone, 
				environment_component=environment,
				spawn_zones_components=spawn_zones,
			)
		elif other_type == 'RandomSpawnYaw':
			from others.randomspawnyaw import RandomSpawnYaw
			other = RandomSpawnYaw(
				drone_component=drone, 
				environment_component=environment,
				yaw_min=0, 
				yaw_max=360,
			)
		elif other_type == 'SpawnEvaluator':
			from others.spawnevaluator import SpawnEvaluator
			other = SpawnEvaluator(
				model_component=model,
				drone_component=drone,
				environment_component=environment,
				evaluate_every_nEpisodes=every_nEpisodes,
				nTimes=4, 
				spawns=([[0,0,start_z],0], [[0,0,start_z],135], [[0,0,start_z],180], [[0,0,start_z],225]),
			)
		elif other_type == 'ModelSaver':
			from others.modelsaver import ModelSaver
			other = ModelSaver(
				model_component=model,
				environment_component=environment,
				save_every_nEpisodes=every_nEpisodes,
			)
		elif other_type == 'ReplayBufferSaver':
			from others.replaybuffersaver import ReplayBufferSaver
			other = ReplayBufferSaver(
				model_component=model,
				environment_component=environment,
				save_every_nEpisodes=every_nEpisodes,
			)
		elif other_type == 'BenchMarker':
			from others.benchmarker import BenchMarker
			other = BenchMarker(
				environment_component=environment,
				benchmark_every_nEpisodes=every_nEpisodes,
			)
		others_components.append(other)

	# ENVIRONMENT
	if environment_type == 'DroneRL':
		from environments.dronerl import DroneRL
		environment = DroneRL(
			drone_component=drone, 
			actor_component=actor, 
			observer_component=observer, 
			rewarder_component=rewarder, 
			terminators_components=terminators_components,
			others_components=others_components,
			name = 'DroneRL'
		)

	# CONTROLLER
	if controller_type == 'Manual':
		from controllers.manual import Manual
		controller = Manual(
			drone_component=drone
		)
	elif controller_type == 'TrainRL':
		from controllers.trainrl import TrainRL
		controller = TrainRL(
			model_component=model,
		)
	elif controller_type == 'EvaluateRL':
		from controllers.evaluaterl import EvaluateRL
		controller = EvaluateRL(
			model_component=model,
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