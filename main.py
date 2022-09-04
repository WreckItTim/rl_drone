import os
import utils
from configuration import Configuration
import math


# USER PARAMETERS and SETUP
# test version is just a name used for logging (optional)
test_version =  'delta'
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
utils.set_global_parameter('working_directory', working_directory)


# META data to log in configuration file - no required format, anything you want to note here
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'timestamp': utils.get_timestamp(),
	'repo_version': 'delta_1',
	'run_name': run_name
	}
# select rather to overwrite meta data in configuration file (only if reading one)
update_meta = False


# create CONTROLLER - controls all components (mode)
# debug mode will prompt user input for which component(s) to debug
if controller_type == 'debug':
	from controllers.debug import Debug
	controller = Debug(
		drone_component='Drone',
		)
# train will create a new or read in a trained model and (continue) train
elif controller_type == 'train':
	from controllers.trainrl import TrainRL
	controller = TrainRL(
		model_component='Model',
		)
# evaluate willl read in a trained model and evaluate on given environment
elif controller_type == 'evaluate':
	from controllers.evaluaterl import EvaluateRL
	controller = EvaluateRL(
		model_component= 'Model',
		)


# READ OLD CONFIGURATION FILE
# you do not need to do nothing else if reading a config file as is
if not make_new_configuration:
	# load configuration file and create object to save and connect components
	configuration = Configuration.load(read_configuration_path)
	if update_meta:
		configuration.update_meta(meta)
	Configuration.set_active(configuration)
	configuration.set_controller(controller)
	
	# ALTER or CREATE any components as desired here


# MAKE NEW CONFIGURATION
# all packaged componets are listed here and created for debugging purposes
# the below configuration is set to run the Delta Demonstration in our paper
# edit as needed, suggested to use as a template
elif make_new_configuration:
	# create new configuration object to save and connect components
	configuration = Configuration(meta)
	Configuration.set_active(configuration)
	configuration.set_controller(controller)
	
	# global parameters to be used by all components
	# name of drone component to define agent commands
	drone = 'AirSim' # AirSim Tello
	# name of observer component to handle the observation space
	observer = 'MultiStack' # Single SingleLag MultiStack
	# names of sensor components to collect observations from environment
	sensors = [
		'Camera', 
		'Distance',
		]
	# observation image height after processing
	output_height = 84 
	# observation image width after processing
	output_width = 84 
	# relative objective point for each episode
	relative_objective_point = (100, 0, 0) 
	# starting height of drone for each episode
	start_z = -4 
	# save and evaluate every n episodes, some model parameters are also a function of this
	every_nEpisodes = 100 
	# drone speed for steps in meters / second
	step_size = 2 
	# drone duration of steps in seconds
	duration = 2 

	# MAP - controls the map that the drone agent will be traversing
	# Microsoft AirSim, simulated map
	if drone == 'AirSim':
		from maps.airsimmap import AirSimMap
		AirSimMap(
			settings = None,
			settings_directory = 'maps/airsim_settings/',
			setting_files = [
				'lightweight', 
				'speedup', 
				'tellocamera', 
				'bellydistance',
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
		duration = duration,
	)
	FixedMove.get_move(
		drone_component = 'Drone', 
		move_type = 'Down', 
		step_size = step_size,
		duration = duration,
	)
	FixedMove.get_move(
		drone_component = 'Drone', 
		move_type = 'Forward', 
		step_size = step_size,
		duration = duration,
	)

	# ACTOR
	from actors.discreteactor import DiscreteActor
	DiscreteActor(
		actions_components=[
			'Down',
			'Forward',
			'Up',
			],
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
		min_distance = 4, 
		max_distance = 104,
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
	from terminators.relativepoint import RelativePoint
	RelativePoint(
		drone_component = 'Drone',
		xyz_point = relative_objective_point,
		min_distance = 4, 
		max_distance = 120,
		name = 'RelativePointTerminator',
	)
	from terminators.rewardthresh import RewardThresh
	RewardThresh(
		rewarder_component='Rewarder',
		min_reward = 0,
		name = 'RewardThresh',
	)
	from terminators.maxsteps import MaxSteps
	MaxSteps(
		max_steps = 50,
		name = 'MaxSteps',
	)

	# MODEL
	if model == 'DQN':
		from models.dqn import DQN
		DQN(
			environment_component = 'TrainEnvironment',
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
			environment_component = 'TrainEnvironment',
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
			environment_component = 'TrainEnvironment',
			policy = 'CnnPolicy',
			learning_rate = 1e-3,
			buffer_size = every_nEpisodes*10,
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
			environment_component = 'TrainEnvironment',
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
			environment_component = 'TrainEnvironment',
			policy = 'CnnPolicy',
			learning_rate = 1e-3,
			buffer_size = every_nEpisodes*10,
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
			environment_component = 'TrainEnvironment',
			policy = 'CnnPolicy',
			learning_rate = 1e-3,
			buffer_size = every_nEpisodes*10,
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

	# SPAWNER
	from others.spawner import Spawner
	from datastructs.spawn import Spawn
	Spawner(
		spawns_components=[
			Spawn(
				x_min=-16,
				x_max=16,
				y_min=-16,
				y_max=16,
				z_min=start_z,
				z_max=start_z,
				yaw_max=2*math.pi,
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
				yaw=math.radians(230),
				random=False,
				),
			],
		name='EvaluateSpawner',
	)

	# EVALUATOR
	from others.evaluator import Evaluator
	Evaluator(
		environment_component = 'EvaluateEnvironment',
		model_component = 'Model',
		frequency = every_nEpisodes,
		nEpisodes = 4,
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
			'RelativePointTerminator',
			'MaxSteps',
			],
		saver_component='Saver',
		evaluator_component='Evaluator',
		spawner_component='TrainSpawner',
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
			'RelativePointTerminator',
			'MaxSteps',
			],
		spawner_component='EvaluateSpawner',
		write_observations=True,
		name = 'EvaluateEnvironment',
	)
utils.speak('configuration created!')


# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# DISCONNECT COMPONENTS
configuration.disconnect_all()

# all done!
utils.speak('Thatll do pig thatll do')