import os
import utils
from configuration import Configuration

# USER PARAMETERS and setup'
repo_version = 'gamma'
test_version =  'debug' # 'debug' 'alpha' 'beta' 'gamma'
model = 'DQN' # DQN A2C DDPG PPO SAC TD3
train_or_evaluate = 'train'
run_name = test_version + '_' + model + '_' + train_or_evaluate
timestamp = utils.get_timestamp() # timestamp used for default write folder, also used to stamp configuration file if write_configuration=True
write_folder = f'temp/' + run_name + '/' # f'temp/{timestamp}/' # will write, by default, all files to this folder
read_configuration_path = 'temp/' + test_version + '_' + model + '_train/configuration.json' # path to read configuration file if read_configuration=True
write_configuration_path = 'temp/' + run_name + '/configuration.json' # path to write configuration file if write_configuration=True
utils.read_global_parameters() # True: sets all variables in the global_parameters.json file
if not os.path.exists('temp/'):
	os.makedirs('temp/')
if not os.path.exists(write_folder):
	os.makedirs(write_folder)
utils.global_parameters['write_folder'] = write_folder # master, default folder to right run log/info to

# force a new configuration even if read_configuration_path exists already
MAKE_NEW = True

# READ OLD CONFIGURATION FILE, you need do nothing else if path is set correctly above
if not MAKE_NEW and os.path.exists(read_configuration_path):
	utils.speak('reading configuration...')
	configuration = Configuration.load(read_configuration_path)
	Configuration.set_active(configuration)
	
	# ALTER ANY READ-IN COMPONENTS or make a new controller as needed here
	if train_or_evaluate == 'evaluate':
		from controllers.evaluaterl import EvaluateRL
		controller = EvaluateRL(
			model_component=model,
			n_eval_episodes=4,
		)
	configuration.set_controller(controller)


# MAKE NEW CONFIGURATION, set parameters, and create component objects one by one by code, as needed below (all packaged components are listed below with __init__ args)
elif MAKE_NEW or not os.path.exists(read_configuration_path):
	utils.speak('creating new configuration...')

	# create configuration object and set active (will add components to this configuration)
	configuration = Configuration(timestamp, repo_version)
	Configuration.set_active(configuration)
	
	# create controller object
	controller_type = 'Debug' # Debug TrainRL EvaluateRL
	# CONTROLLER
	if controller_type == 'Debug':
		from controllers.debug import Debug
		controller = Debug(
		drone_component='Drone'
	)
	elif controller_type == 'TrainRL':
		from controllers.trainrl import TrainRL
		controller = TrainRL(
		model_component='Model',
	)
	elif controller_type == 'EvaluateRL':
		from controllers.evaluaterl import EvaluateRL
		controller = EvaluateRL(
		model_component='Model',
		n_eval_episodes=4,
	)
	configuration.set_controller(controller)
	
	# global parameters to be used by all components
	drone = 'AirSim' # AirSim Tello
	observer = 'MultiStack' # Single SingleLag MultiStack
	sensors = [
		'Camera', 
		'Distance',
		]
	output_height = 84 # output shape after processing
	output_width = 84 # output shape after processing
	relative_objective_point = (100, 0, 0)
	start_z = -4
	every_nEpisodes = 2
	step_size = 4 # meters
	speed = 4 # meters / second

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
				'bellydistance',
				'nodisplay',
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
		actions_components=[
#			'Up',
#			'Down',
			'Forward',
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
		rewarder_component='Rewarder',
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
	from others.spawner import Spawner
	from datastructs.spawn import Spawn
	Spawner(
		Spawn(
			x_min=-10, 
			x_max=10,
			y_min=-10, 
			y_max=10, 
			z_min=start_z, 
			z_max=start_z,
			yaw_max=360,
			random=True,
			),
		spawn_on_train=True,
		spawn_on_evaluate=False,
		name='Spawner',
	)
	from others.evaluator import Evaluator
	Evaluator(
		model_component='Model',
		drone_component='Drone',
		environment_component='Environment',
		spawners_components=[
			Spawner(
				Spawn(
					z=start_z, 
					random=False,
				),
			),
			Spawner(
				Spawn(
					z=start_z, 
					yaw=135, 
					random=False,
				),
			),
			Spawner(
				Spawn(
					z=start_z, 
					yaw=180, 
					random=False,
				),
			),
			Spawner(
				Spawn(
					z=start_z, 
					yaw=225, 
					random=False,
				),
			),
			],
		evaluate_every_nEpisodes=every_nEpisodes, 
		_write_folder=None, 
		nEvaluations=0,
		name='Evaluator',
	)
	from others.saver import Saver
	Saver(
		model_component='Model', 
		environment_component='Environment',
		nEpisodes=0, 
		save_every_nEpisodes=every_nEpisodes, 
		save_model=True,
		save_replay_buffer=True,
		save_configuration_file=True,
		save_benchmarks=True,
		_write_folder=None,
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
			#'RewardThresh',
			'MaxSteps',
			],
		others_components=[
			'Spawner',
			'Evaluator',
			'Saver',
			],
		name = 'Environment'
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