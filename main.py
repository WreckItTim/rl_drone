import os
from utils import *
from component import *


# USER PARAMETERS
timestamp = get_timestamp()
write_folder = f'temp/{timestamp}/'
if not os.path.exists(write_folder):
	os.makedirs(write_folder)
read_configuration = False
read_configuration_path = 'configurations/overwrite_configuration.json'
write_configuration = True
write_configuration_path = write_folder + '/overwrite_configuration.json'
read_global_parameters()
global_parameters['write_folder'] = write_folder


# READ OLD CONFIGURATION FILE ??
if read_configuration:
	configuration = read_json(read_configuration_path)
	controller, components, timestamp = deserialize_configuration(configuration)

	# OVERWRITE ANY CONFIGURATION ARGUMENTS HERE


# OR MAKE NEW CONFIGURATION ??
else:
	# specify components to be used
	map_type = 'AirSim' # AirSim Field
	drone_type = 'AirSim' # AirSim Tello
	sensor_type = 'AirSimCamera' # AirSimCamera PortCamera
	transformer_types = ['ResizeImage', 'NormalizeDepth'] # ResizeImage NormalizeDepth
	observer_type = 'Single' # Single
	action_type = 'FixedMove' # FixedMove
	move_types = ['up', 'down', 'forward'] # up down forward backward left right (any combination-diagnol by using an '_' to seperate)
	actor_type = 'DiscreteActor' # DiscreteActor ContinuousActor
	reward_types = ['Avoid', 'RelativePoint'] # Avoid RelativePoint
	rewarder_type = 'Schema' # Schema
	terminator_types = ['Collision', 'RelativePoint', 'MaxSteps'] # Collision RelativePoint RewardThresh MaxSteps
	other_types = ['RandomSpawnPoint', 'RandomSpawnYaw', 'SpawnEvaluator', 'ModelSaver'] # RandomSpawnPoint RandomSpawnYaw ForwardEvaluator ModelSaver
	environment_type = 'DroneRL' # DroneRL
	model_type = 'DQN' # A2C DDPG DQN PPO SAC TD3
	controller_type = 'TrainRL' # TrainRL EvaluateRL Manual

	# specify other parameters to be used by components
	image_shape=(84, 84, 1)
	objective_point=(100, 0, 0)
	from datastructs.zone import Zone
	spawn_zones = [
		Zone(x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=0, z_max=0),
	]


	# MAP
	if map_type == 'AirSim':
		from maps.airsimmap import AirSimMap
		map_ = AirSimMap(
			settings=None,
			setting_files=['base'],
			release_file='Blocks',
			release_directory='resources/airsim_maps/',
			settings_directory='resources/airsim_settings/',
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
	transformer_components=[] # robots in disguise! 
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
		transformer_components.append(transformer)

	# OBSERVER
	if observer_type == 'Single':
		from observers.single import Single
		observer = Single(
			sensor_component=sensor, 
			transformer_components=transformer_components,
			please_write=True, 
			write_directory='temp/',
			output_shape=image_shape,
		)

	# ACTION
	action_components = []
	if action_type == 'FixedMove':
		from actions.fixedmove import FixedMove 
		for move_type in move_types:
			fixed_move = FixedMove.get_move(
				drone_component=drone, 
				move_type=move_type, 
				step_size=5,
				speed=4,
			)
			action_components.append(fixed_move)

	# ACTOR
	if actor_type == 'DiscreteActor':
		from actors.discreteactor import DiscreteActor
		actor = DiscreteActor(
			action_components=action_components,
		)

	# REWARD
	reward_components=[]
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
				xyz_point = objective_point,
				min_distance = 5,
				max_distance = 110,
			)
		reward_components.append(reward)

	# REWARDER
	if rewarder_type == 'Schema':
		from rewarders.schema import Schema
		rewarder = Schema(
			reward_components=reward_components,
			reward_weights=[1, 2],
		)

	# TERMINATOR
	terminator_components=[]
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
				xyz_point = objective_point,
				min_distance = 5,
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
				max_steps = 40,
			)
		terminator_components.append(terminator)

	# MODEL
	if model_type == 'DQN':
		from models.dqn import DQN
		model = DQN(
			policy = 'CnnPolicy',
			learning_rate = 1e-4,
			buffer_size = 1_000_000,
			learning_starts = 50000,
			batch_size = 32,
			tau = 1.0,
			gamma = 0.99,
			train_freq = 4,
			gradient_steps = 1,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			target_update_interval = 10000,
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
		)
	elif model_type == 'A2C':
		from models.a2c import A2C
		model = A2C(
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
		)
	elif model_type == 'DDPG':
		from models.ddpg import DDPG
		model = DDPG(
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
		)
	elif model_type == 'PPO':
		from models.ppo import PPO
		model = PPO(
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
		)
	elif model_type == 'SAC':
		from models.sac import SAC
		model = SAC(
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
		)
	elif model_type == 'TD3':
		from models.td3 import TD3
		model = TD3(
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
		)

	# OTHER
	other_components = []
	for other_type in other_types:
		other = None
		if other_type == 'RandomSpawnPoint':
			from others.randomspawnpoint import RandomSpawnPoint
			other = RandomSpawnPoint(
				drone_component=drone, 
				spawn_zone_components=spawn_zones,
			)
		elif other_type == 'RandomSpawnYaw':
			from others.randomspawnyaw import RandomSpawnYaw
			other = RandomSpawnYaw(
				drone_component=drone, 
				yaw_min=0, 
				yaw_max=360,
			)
		elif other_type == 'SpawnEvaluator':
			from others.spawnevaluator import SpawnEvaluator
			other = SpawnEvaluator(
				model_component=model,
				drone_component=drone,
				evaluate_every_nEpisodes=10,
				nTimes=3, 
				spawns=([[0,0,0],0], [[0,0,0],120], [[0,0,0],240]),
			)
		elif other_type == 'ModelSaver':
			from others.modelsaver import ModelSaver
			other = ModelSaver(
				model_component=model,
				save_every_nEpisodes=5,
			)
		other_components.append(other)

	# ENVIRONMENT
	if environment_type == 'DroneRL':
		from environments.dronerl import DroneRL
		environment = DroneRL(
			drone_component=drone, 
			actor_component=actor, 
			observer_component=observer, 
			rewarder_component=rewarder, 
			terminator_components=terminator_components,
			other_components=other_components,
		)
	model.set(environment)

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
		from controllers.trainrl import EvaluateRL
		controller = EvaluateRL(
			model_component=model,
		)
print('configuration loaded!')


# FETCH COMPONENTS
components = get_all_components()


# WRITE CONFIGURATION
if write_configuration:
	configuration = serialize_configuration(controller, components, timestamp)
	write_json(configuration, write_configuration_path)
write_global_parameters()
 

# CONNECT COMPONENTS
for component in components:
	component.connect()
print('components connected!')
write_global_parameters()


# RUN CONTROLLER
print('running controller...')
controller.run()
write_global_parameters()


# LOG BENCHMARKS
components = get_all_components()
for component in reversed(components):
	log_memory(component)
write_json(benchmarks, write_folder + 'benchmarks.json')
write_global_parameters()


# DISCONNECT COMPONENTS
for component in reversed(components):
	component.disconnect()
print('components disconnected!')
write_global_parameters()

# ALL DONE
print('Good bye!')