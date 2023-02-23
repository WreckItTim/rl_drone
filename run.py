import utils
from configuration import Configuration
import math
import sys
repo_version = 'gamma10'

# ADJUST REPLAY BUFFER SIZE PENDING AVAILABLE RAM see replay_buffer_size bellow

# grab arguments input from terminal
args = sys.argv
# first sys argument is test_case to run (see options below)
	# if no arguments will default to test_case 'H4' - blocks, horizontal motion, with an MLP
test_case = 'H4'
if len(args) > 1:
	test_case = args[1].upper()
# third sys argument is any text to concatenate to run output folder name (i.e. run2 etc)
	# will assume no text to concat if no additional input
run_post = ''
if len(args) > 2:
	run_post = args[2]
# second sys argument is to continue training from last checkpoint (True) or not (False)
	# will assume False if no additional input
	# must pass in run_post var
continue_training = False
if len(args) > 3:
	continue_training = args[3] in ['true', 'True']
	
# airsim map to use?
airsim_release = 'Blocks'
if test_case in ['M9', 'S1']:
	airsim_release = 'AirSimNH'
if test_case in ['PC']:
	airsim_release = 'CityEnviron'

# unlock vertical motion?
vert_motion = False
if test_case in ['H3', 'M9', 'TB']:
	vert_motion = True

# MLP or CNN?
policy = 'MultiInputPolicy' # CNN (2d depth map)
if test_case in ['H3', 'H4', 'S2']:
	policy = 'MlpPolicy' # MLP (flattened depth map)

# TD3 or DQN?
rl_model = 'TD3'
if test_case in ['S2']:
	rl_model = 'DQN'

replay_buffer_size = 400_000 # 400_000 will work well within a 32gb-RAM system when using MultiInputPolicy
							 # if using an MlpPolicy this will use drastically less memory

# see bottom of this file which calls functions to create components and run controller

# runs some overarching base things
def create_base_components(
		airsim_release = 'Blocks', # name of airsim release to use, see maps.arisimmap
		vert_motion = False, # allowed to move on z-axis? False will restrict motion to horizontal plane
		policy = 'MlpPolicy', # MultiInputPolicy MlpPolicy - which neural net for RL model to use 
		rl_model = 'TD3', # which SB3 RL model to use - TD3 DQN (see models folder for others)
		replay_buffer_size = 1_000_000, # a size of 1_000_000 requires 56.78 GB if using MultiInputPolicy
		continue_training = False, # set to true if continuing training from checkpoint
		controller_type = 'Train', # Train, Debug, Drift, Evaluate
		actor = 'Teleporter', # Teleporter Continuous
		clock_speed = 10, # airsim clock speed (increasing this will also decerase sim-quality)
		training_steps = 50_000_000, # max number of training steps 
		max_distance = 100, # distance contraint used for several calculations (see below)
		nTimesteps = 4, # number of timesteps to use in observation space
		checkpoint = 100, # evaluate model and save checkpoint every # of episodes
		run_post = '', # optionally add text to generated run name (such as run2, retry, etc...)
):

	# **** SETUP ****
	run_name = airsim_release + '_' + str(vert_motion) + '_' + policy
	if run_post != '':
		run_name += '_' + run_post

	# get OS, set file IO paths
	OS = utils.setup(
		write_parent = 'local/runs/',
		run_prefix = repo_version + '_' + run_name,
		)
	working_directory = utils.get_global_parameter('working_directory')

	# CREATE CONTROLLER
	controller = utils.get_controller(
		controller_type = controller_type,
		total_timesteps = training_steps, # optional if using train - all other hypers set from model instance
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
		# get airsim release to launch
		release_path = None
		if utils.get_global_parameter('OS') == 'windows':
			if airsim_release == 'Blocks':
				release_path = 'local/airsim_maps/Blocks/WindowsNoEditor/Blocks.exe'
			if airsim_release == 'AirSimNH':
				release_path = 'local/airsim_maps/AirSimNH/WindowsNoEditor/AirSimNH.exe'
			if airsim_release == 'CityEnviron':
				release_path = 'local/airsim_maps/CityEnviron/WindowsNoEditor/CityEnviron.exe'
		if utils.get_global_parameter('OS') == 'linux':
			if airsim_release == 'Blocks':
				release_path = 'local/airsim_maps/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
			if airsim_release == 'AirSimNH':
				release_path = 'local/airsim_maps/AirSimNH/LinuxNoEditor/AirSimNH.sh'
		# add console flags
		console_flags = []
		# render screen? This should be false if SSH-ing from remote
		render_screen = utils.get_global_parameter('render_screen')
		if render_screen:
			console_flags.append('-Windowed')
		else:
			console_flags.append('-RenderOffscreen')
		# create airsim map object
		AirSimMap(
			voxels_component='Voxels',
			release_path = release_path,
			settings = {
				'ClockSpeed': clock_speed,
				},
			setting_files = [
				'lightweight', # see maps/airsim_settings
				],
			console_flags = console_flags.copy(),
			name = 'Map',
		)
		# voxels grabs locations of objects from airsim map
		# used to validate spawn and goal points (not inside an object)
		# also used to visualize flight paths
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
		x_bounds = [-1*max_distance, max_distance]
		y_bounds = [-1*max_distance, max_distance]
		z_bounds = [-4, -4]
		# dynamic goal will spawn in bounds - randomly for train, static for evaluate
		# goal distance will increase, "amp up", with curriculum learning
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
		# heavy penalty for collision
		from rewards.collision import Collision
		Collision(
			drone_component = 'Drone',
			name = 'CollisionReward',
		)
		# increasing reward as approaches goal
		from rewards.goal import Goal
		Goal(
			drone_component = 'Drone',
			goal_component = 'Goal',
			include_z = True if vert_motion else False, # includes z in distance calculations
			name = 'GoalReward',
		)
		# penalize heavier as approaches time constraint
		from rewards.steps import Steps
		Steps(
			name = 'StepsReward',
			max_steps = 4**(1+vert_motion), # base number of steps, will scale with further goal
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
				2,
				2,
				1,
			],
			name = 'Rewarder',
		)

		# ACTIONS
		if rl_model in ['TD3']:
			base_distance = 10 # meters, will multiply rl_output by this value
			base_yaw = math.pi # degrees, will multiply rl_output by this value
			from actions.move import Move 
			Move(
				drone_component = 'Drone', 
				base_x_rel = base_distance, 
				name = 'MoveForward',
			)
			from actions.rotate import Rotate 
			Rotate(
				drone_component = 'Drone',  
				base_yaw = base_yaw,
				min_space = -1, # allows left and right rotations
				name = 'Rotate',
			)
			actions = [
				'MoveForward',
				'Rotate'
			]
			if vert_motion:
				Move(
					drone_component = 'Drone', 
					base_z_rel = base_distance, 
					min_space = -1, # allows up and down movements
					name = 'MoveVertical',
				)
				actions.append('MoveVertical')
		if rl_model in ['DQN']:
			from actions.fixedmove import FixedMove 
			FixedMove(
				drone_component = 'Drone', 
				x_speed = 1, 
				name = 'FixedForward1',
			)
			FixedMove(
				drone_component = 'Drone', 
				x_speed = 5, 
				name = 'FixedForward2',
			)
			FixedMove(
				drone_component = 'Drone', 
				x_speed = 10, 
				name = 'FixedForward3',
			)
			from actions.fixedrotate import FixedRotate 
			FixedRotate(
				drone_component = 'Drone',  
				yaw_rate = math.pi / 16,
				name = 'FixedRotate1',
			)
			FixedRotate(
				drone_component = 'Drone',  
				yaw_rate = math.pi / 8,
				name = 'FixedRotate2',
			)
			FixedRotate(
				drone_component = 'Drone',  
				yaw_rate = math.pi / 2,
				name = 'FixedRotate3',
			)
			FixedRotate(
				drone_component = 'Drone',  
				yaw_rate = -1 * math.pi / 16,
				name = 'FixedRotate4',
			)
			FixedRotate(
				drone_component = 'Drone',  
				yaw_rate = -1 * math.pi / 8,
				name = 'FixedRotate5',
			)
			FixedRotate(
				drone_component = 'Drone',  
				yaw_rate = -1 * math.pi / 2,
				name = 'FixedRotate6',
			)
			actions = [
				'FixedForward1',
				'FixedForward2',
				'FixedForward3',
				'FixedRotate1',
				'FixedRotate2',
				'FixedRotate3',
				'FixedRotate4',
				'FixedRotate5',
				'FixedRotate6',
			]
			if vert_motion:
				FixedMove(
					drone_component = 'Drone', 
					z_speed = -1, 
					name = 'FixedUp1',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = -5, 
					name = 'FixedUp2',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = -10, 
					name = 'FixedUp3',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = 1, 
					name = 'FixedDown1',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = 5, 
					name = 'FixedDown2',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = 10, 
					name = 'FixedDown3',
				)
				actions.append('FixedUp1')
				actions.append('FixedUp2')
				actions.append('FixedUp3')
				actions.append('FixedDown1')
				actions.append('FixedDown2')
				actions.append('FixedDown3')

		# ACTOR
		if rl_model in ['TD3']:
			if actor == 'Continuous':
				from actors.continuousactor import ContinuousActor
				ContinuousActor(
					actions_components = actions,
					name='Actor',
				)
			if actor == 'Teleporter':
				print('teleporter ACTIVE')
				from actors.teleporter import Teleporter
				Teleporter(
					drone_component = 'Drone',
					actions_components = actions,
					name='Actor',
				)
		if rl_model in ['DQN']:
			from actors.discreteactor import DiscreteActor
			DiscreteActor(
				actions_components = actions,
				name='Actor',
			)

		# CREATE OBSERVATION SPACE
		# TRANSFORMERS
		from transformers.gaussiannoise import GaussianNoise
		GaussianNoise(
			deviation = 0.5,
			name = 'PositionNoise',
		)
		GaussianNoise(
			deviation = math.radians(5),
			name = 'OrientationNoise',
		)
		from transformers.gaussianblur import GaussianBlur
		GaussianBlur(
			sigma = 2,
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
		# SENSORS
		# keep track of recent past actions
		from sensors.actions import Actions
		Actions(
			actor_component = 'Actor',
			name = 'ActionsSensor',
			)
		# keep track of time steps
		from sensors.steps import Steps
		Steps(
			steps_component = 'StepsReward',
			name = 'StepsSensor',
			)
		# sense linear distance to goal
		from sensors.distance import Distance
		Distance(
			misc_component = 'Drone',
			misc2_component = 'Goal',
			include_z = False,
			prefix = 'drone_to_goal',
			transformers_components = [
				#'PositionNoise',
				'NormalizeDistance',
				], 
			name = 'GoalDistance',
		)
		# sense yaw difference to goal 
		from sensors.orientation import Orientation
		Orientation(
			misc_component = 'Drone',
			misc2_component = 'Goal',
			prefix = 'drone_to_goal',
			transformers_components = [
				#'OrientationNoise',
				'NormalizeOrientation',
				],
			name = 'GoalOrientation',
		)
		if vert_motion:
			# sense altitude distance to goal
			Distance(
				misc_component = 'Drone',
				misc2_component = 'Goal',
				include_x = False,
				include_y = False,
				prefix = 'drone_to_goal',
				transformers_components = [
					#'PositionNoise',
					'NormalizeDistance',
					],
				name = 'GoalAltitude',
			)
		from sensors.airsimcamera import AirSimCamera
		if policy == 'MultiInputPolicy':
			# get 2d depth map from camera
			AirSimCamera(
				airsim_component = 'Map',
				transformers_components = [
					'ResizeImage',
					#'DepthNoise',
					'NormalizeDepth',
					],
				name = 'DepthMap',
				)
		if policy == 'MlpPolicy':
			# get flattened depth map (obsfucated front facing distance sensors)
			from transformers.resizeflat import ResizeFlat
			max_cols = [16, 32, 52, 68, 84] # splits depth map by columns
			max_rows = [21, 42, 63, 84] if vert_motion else [42] # splits depth map by rows
			ResizeFlat(
				max_cols = max_cols,
				max_rows = max_rows,
				name = 'ResizeFlat',
			)
			AirSimCamera(
				airsim_component = 'Map',
				transformers_components = [
					'ResizeImage',
					#'DepthNoise',
					'NormalizeDistance',
					'ResizeFlat',
					],
				name = 'FlattenedDepth',
				)

		# OBSERVER
		vector_sensors = ['ActionsSensor', 'StepsSensor', 'GoalDistance', 'GoalOrientation']
		if rl_model in ['DQN']:
			vector_length = 1 + 1 + 1 + 1
		if rl_model in ['TD3']:
			vector_length = len(actions) + 1 + 1 + 1
		if vert_motion:
			vector_sensors.append('GoalAltitude')
			vector_length += 1
		if policy == 'MlpPolicy':
			vector_sensors.append('FlattenedDepth')
			vector_length += len(max_cols) * len(max_rows)
		print('ACTIONS', actions)
		print('VECTOR_LENGTH', vector_length)
		from observers.single import Single
		Single(
			sensors_components = vector_sensors, 
			vector_length = vector_length,
			nTimesteps = nTimesteps,
			name = 'Observer' if policy == 'MlpPolicy' else 'ObserverVector',
		)
		if policy == 'MultiInputPolicy':
			Single(
				sensors_components = ['DepthMap'], 
				is_image = True,
				image_height = 84, 
				image_width = 84,
				image_bands = 1,
				nTimesteps = nTimesteps,
				name = 'ObserverImage',
			)
			from observers.multi import Multi
			Multi(
				vector_observer_component = 'ObserverVector',
				image_observer_component = 'ObserverImage',
				name = 'Observer',
				)

		# MODEL
		if rl_model == 'TD3':
			from models.td3 import TD3
			TD3(
				environment_component = 'TrainEnvironment',
				policy = policy,
				buffer_size = replay_buffer_size,
				tensorboard_log = working_directory + 'tensorboard_log/',
				name='Model',
			)
		if rl_model == 'DQN':
			from models.dqn import DQN
			DQN(
				environment_component = 'TrainEnvironment',
				policy = policy,
				buffer_size = replay_buffer_size,
				tensorboard_log = working_directory + 'tensorboard_log/',
				name='Model',
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
					x_min=-max_distance,
					x_max=max_distance,
					y_min=-max_distance,
					y_max=max_distance,
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
		'''
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
		'''
		if not vert_motion and actor != 'Teleporter':
			# ALTITUDE ADJUSTER (for horizontal motion, 
				# since moving forward naturally adds upward drift up)
			from modifiers.altadjust import AltAdjust
			AltAdjust(
				base_component = 'Actor',
				parent_method = 'step',
				drone_component = 'Drone',
				order = 'post',
				name = 'Evaluator',
			)

	return configuration


def run_controller(configuration):
	utils.speak('configuration created!')

	# CONNECT COMPONENTS
	configuration.connect_all()

	# view neural net archetecture
	model_name = str(configuration.get_component('Model')._child())
	sb3_model = configuration.get_component('Model')._sb3model
	print('MODEL NAME', model_name)
	if 'dqn' in model_name:
		print(sb3_model.q_net)
		for name, param in sb3_model.q_net.named_parameters():
			msg = str(name) + ' ____ ' + str(param[0])
			utils.speak(msg)
			break
	if 'td3' in model_name:
		print(sb3_model.critic)
		for name, param in sb3_model.critic.named_parameters():
			msg = str(name) + ' ____ ' + str(param[0])
			utils.speak(msg)
			break
	utils.speak('all components connected. Send any key to continue...')
	x = input()

	# WRITE CONFIGURATION
	configuration.save()

	# RUN CONTROLLER
	configuration.controller.run()

	# done
	configuration.controller.stop()

# create base components
configuration = create_base_components(
		airsim_release = airsim_release, # name of airsim release to use, see maps.arisimmap
		vert_motion = vert_motion, # allowed to move on z-axis? False will restrict motion to horizontal plane
		policy = policy, # MultiInputPolicy MlpPolicy - which neural net for RL model to use 
		rl_model = rl_model, # which SB3 RL model to use - TD3 DQN (see models folder for others)
		continue_training = continue_training, # set to true if continuing training from checkpoint
		replay_buffer_size = replay_buffer_size,
		run_post = run_post, # optionally add text to generated run name (such as run2, retry, etc...)
)

# create any other components
if not continue_training:
	# add components here
	pass

# run baby run
run_controller(configuration)
