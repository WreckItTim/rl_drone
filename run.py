import rl_utils as utils
from configuration import Configuration
import math
import numpy as np
import sys
import os
from hyperopt import hp
repo_version = 'gamma21'

# ADJUST REPLAY BUFFER SIZE PENDING AVAILABLE RAM see replay_buffer_size bellow

# grab arguments input from terminal
args = sys.argv
# first sys argument is test_case to run (see options below)
	# if no arguments will default to test_case 'H4' - blocks, horizontal motion, with an MLP
test_case = ''
if len(args) > 1:
	test_case = args[1].lower()
# second sys argument is to continue training from last checkpoint (True) or not (False)
	# will assume False if no additional input
	# must pass in run_post var
continue_training = False
if len(args) > 2:
	continue_training = args[2] in ['true', 'True']
# third sys argument is any text to concatenate to run output folder name (i.e. run2 etc)
	# will assume no text to concat if no additional input
run_post = ''
if len(args) > 3:
	run_post = args[3]

# airsim map to use?
airsim_release = 'Blocks'
if test_case in []:
	airsim_release = 'AirSimNH'
if test_case in ['pc']:
	airsim_release = 'CityEnviron'
if test_case in []:
	airsim_release = 'Tello'

action_noise = None
if test_case in ['h3', 'h4', 'tp']:
	action_noise = 'normal'

include_resolution = True
if test_case in ['pc']:
	include_resolution = False

# steps distance collision goal max_steps
reward_weights = [0, 1, 120, 240, 0]
if test_case in ['h3', 's1']:
	reward_weights = [0, 2, 220, 440, 0]
if include_resolution:
	# res1 res2 
	if test_case in ['h4', 's2']:
		reward_weights = [0.25, 0.25] + reward_weights
	else:
		reward_weights = [0.5, 0.5] + reward_weights

# unlock vertical motion?
vert_motion = True
if test_case in []:
	vert_motion = False

# MLP or CNN?
policy = 'MlpPolicy' # MLP (flattened depth map)
if test_case in []:
	policy = 'MultiInputPolicy' # CNN (2d depth map)

# TD3 or DQN?
rl_model = 'TD3'
if test_case in []:
	rl_model = 'DQN'

# read model and/or replay buffer?
read_model_path = None
read_replay_buffer_path = None
if test_case in []:
	read_model_path = 'local/models/GAMMA_model.zip'
	run_post += '_GAMMA'
	#read_replay_buffer_path = 'local/models/GAMMA_replay_buffer.zip'
if test_case in []:
	read_model_path = 'local/models/GAMMA2_model.zip'
	run_post += '_GAMMA2'
if test_case in []:
	read_model_path = 'local/models/DELTA_model.zip'
	run_post += '_DELTA'
if test_case in []:
	read_model_path = 'local/models/DELTA2_model.zip'
	run_post += '_DELTA2'
	#read_replay_buffer_path = 'local/models/DELTA_replay_buffer.zip'
if test_case in []:
	read_model_path = 'local/models/EPSILON_model.zip'
	run_post += '_EPSILON'
	#read_replay_buffer_path = 'local/models/EPSILON_replay_buffer.zip'

# hyper parameter search?
hyper = False
if test_case in []:
	hyper = True
if hyper:
	run_post += '_hyper'

# which hypers to explore?
hyper_params = []
if test_case in []:
	hyper_params.append('learning_rate')
if test_case in []:
	hyper_params.append('learning_starts')
if test_case in []:
	hyper_params.append('buffer_size')
if test_case in []:
	hyper_params.append('tau')
if test_case in []:
	hyper_params.append('batch_size')
if test_case in []:
	hyper_params.append('train_freq')
if test_case in []:
	hyper_params.append('policy_delay')
if test_case in []:
	hyper_params.append('target_policy_noise')
if test_case in []:
	hyper_params.append('target_noise_clip')
if test_case in []:
	hyper_params.append('policy_layers')
if test_case in []:
	hyper_params.append('policy_nodes')

# how may previous steps to train on
replay_buffer_size = 400_000 # 400_000 will work well within a 32gb-RAM system when using MultiInputPolicy
							 # if using an MlpPolicy this will use drastically less memory

# after how many steps to stop training
training_steps = 1_000_000 # roughly 250k steps a day
if test_case in []:
	training_steps = 40_000 # hyper surrogate model size

flat = 'big2'
if test_case in []:
	flat = 'small'
if test_case in []:
	flat = 'big'

distance_reward = 'exp2'
if test_case in []:
	distance_reward = 'exp'

step_reward = 'none'
if test_case in []:
	step_reward = 'scale2'

learning_starts = 100
if test_case in []:
	learning_starts = 500

# see bottom of this file which calls functions to create components and run controller
controller_type = 'Train' # Train, Debug, Drift, Evaluate
if test_case in []:
	controller_type = 'Debug'
actor = 'Teleporter' # Teleporter Continuous
if test_case in []:
	actor = 'Continuous'
clock_speed = 10 # airsim clock speed (increasing this will also decerase sim-quality)
# office-lab 35x22 tiles which are 30x30 cm squares, 10.5 max meters
# halls... h1:5x14 h2:5x60 h3:5x76 l1:13x19 h4:6x22, 22.8 max meters
max_distance = 100 # distance contraint used for several calculations (see below)
if test_case in []:
	max_distance = 25
tello_goal = ''
if test_case in []:
	tello_goal = 'Hallway1'
adjust_for_yaw = True
if test_case in []:
	adjust_for_yaw = False

include_bottom = True
if test_case in []:
	include_bottom = False


nTimesteps = 4 # number of timesteps to use in observation space
checkpoint = 100 # evaluate model and save checkpoint every # of episodes

# runs some overarching base things
def create_base_components(
		airsim_release = 'Blocks', # name of airsim release to use, see maps.arisimmap
		vert_motion = False, # allowed to move on z-axis? False will restrict motion to horizontal plane
		policy = 'MlpPolicy', # MultiInputPolicy MlpPolicy - which neural net for RL model to use 
		rl_model = 'TD3', # which SB3 RL model to use - TD3 DQN (see models folder for others)
		hyper = False,
		replay_buffer_size = 1_000_000, # a size of 1_000_000 requires 56.78 GB if using MultiInputPolicy
		continue_training = False, # set to true if continuing training from checkpoint
		controller_type = 'Train', # Train, Debug, Drift, Evaluate
		actor = 'Teleporter', # Teleporter Continuous
		clock_speed = 10, # airsim clock speed (increasing this will alsovalue_typevalue_typee
		checkpoint = 100, # evaluate model and save checkpoint every # of episodes
		flat = 'big', # determines size of flattened depth sensor array 
		distance_reward = 'scale2', # # reward function that penalizes distance to goal (large positive fore reaching)
		step_reward = 'scale2', # reward function that penalizes longer episode length
		run_post = '', # optionally add text to generated run name (such as run2, retry, etc...)
		hyper_params = [], # which hyper parameters to hyper tune if model is type hyper
		read_model_path = None, # load pretrained model?
		read_replay_buffer_path = None, # load prebuilt replay buffer?
		include_d = True, # inculde little d=distance/start_distance in sensors
		reward_weights = [1,1,1], # reward weights in order: goal, collision, steps
		learning_starts = 100, # how many steps to collect in buffer before training starts
		tello_goal = '',
		adjust_for_yaw = True,
		include_resolution = True,
		include_bottom = False,
		training_steps = 50_000_000,
		max_distance = 100,
		nTimesteps = 4,
		action_noise = None,
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
		use_wandb = True, # logs tensor board and wandb
		log_interval = 10,
		evaluator = 'Evaluator',
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
			model_component='Model',
			name='TrainEnvironment',
		)
		# CREATE EVALUATE ENVIRONMENT
		GoalEnv(
			drone_component='Drone', 
			actor_component='Actor', 
			observer_component='Observer', 
			rewarder_component='Rewarder', 
			goal_component='Goal',
			model_component='Model',
			is_evaluation_env=True,
			name='EvaluateEnvironment',
		)
		

		# CREATE MAP
		if airsim_release == 'Tello':
			from maps.field import Field
			Field(
				name = 'Map',
			)
		else:
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

		# Create bounds to spawn in and for goal
		from others.bounds import Bounds
		training_bounds = Bounds(
					center = [-20, 0, 0],
					inner_radius = 20,
					outter_radius = 200,
					min_z = -4,
					max_z = -4,
					name = 'TrainingBounds'
					)
		goal_bounds = Bounds(
					center = [-20, 0, 0],
					inner_radius = 0,
					outter_radius = 200,
					min_z = -100,
					max_z = 0,
					name = 'GoalBounds'
					)


		# CREATE DRONE
		if airsim_release == 'Tello':
			from drones.tello import Tello
			Tello(
				name = 'Drone',
			)
		else:
			from drones.airsimdrone import AirSimDrone
			AirSimDrone(
				airsim_component = 'Map',
				name='Drone',
			)

		
		from others.relativegoal import RelativeGoal
		if airsim_release == 'Tello':
			# office-lab 35x22 tiles which are 30x30 cm squares, 10.5 max meters
			# halls... h1:5x14 h2:5x60 h3:5x76 l1:13x19 h4:6x22, 22.8 max meters
			# origin at (3 3) - offset 3 tiles from walls in hallway 1 corner
			if tello_goal == 'Hallway1':
				RelativeGoal(
					drone_component = 'Drone',
					map_component = 'Map',
					static_point = [2.7, 0, 0],
					name = 'Goal',
					)
			if tello_goal == 'Hallway2':
				RelativeGoal(
					drone_component = 'Drone',
					map_component = 'Map',
					static_point = [2.7, -16.8, 0],
					name = 'Goal',
					)
			if tello_goal == 'Hallway3':
				RelativeGoal(
					drone_component = 'Drone',
					map_component = 'Map',
					static_point = [-19.6, -16.8, 0],
					name = 'Goal',
					)
			if tello_goal == 'hallway4':
				RelativeGoal(
					drone_component = 'Drone',
					map_component = 'Map',
					static_point = [-20.8, -10.2, 0],
					name = 'Goal',
					)
		else:
			# CREATE GOAL
			# dynamic goal will spawn in bounds - randomly for train, static for evaluate
			# goal distance will increase, "amp up", with curriculum learning
			RelativeGoal(
				drone_component = 'Drone',
				map_component = 'Map',
				bounds_component = 'GoalBounds',
				static_r = 6, # relative distance for static goal from drone
				static_dz = 4, # relative z for static goal from drone (this is dz above roof or floor)
				static_yaw = 0, # relative yaw for static goal from drone
				random_r = [6,8], # relative distance for random goal from drone
				random_dz = [4,4], # relative z for random goal from drone (this is dz above roof or floor)
				random_yaw = [-1*np.pi, np.pi], # relative yaw for random goal from drone
				random_point_on_train = True, # random goal when training?
				name = 'Goal',
			)

		# CREATE REWARDS AND TERMINATORS

		# REWARDS
		rewards = []
		# penalty for higher resolutions
		if include_resolution:
			from rewards.resolution import Resolution
			Resolution(
				resolution_component = 'FlattenedDepthResolution',
				name = 'ResolutionReward',
			)
			rewards.append('ResolutionReward')
			if include_bottom:
				Resolution(
					resolution_component = 'FlattenedDepthResolution2',
					name = 'ResolutionReward2',
				)
				rewards.append('ResolutionReward2')
		# penalize heavier as approaches time constraint
		from rewards.steps import Steps
		Steps(
			name = 'StepsReward',
			value_type = step_reward,
		)
		rewards.append('StepsReward')
		# increasing reward as approaches goal
		from rewards.distance import Distance
		Distance(
			drone_component = 'Drone',
			goal_component = 'Goal',
			value_type = distance_reward,
			include_z = True if vert_motion else False, # includes z in distance calculations
			name = 'DistanceReward',
		)
		rewards.append('DistanceReward')
		# heavy penalty for collision
		from rewards.collision import Collision
		Collision(
			drone_component = 'Drone',
			name = 'CollisionReward',
		)
		rewards.append('CollisionReward')
		# increasing reward as approaches goal
		from rewards.goal import Goal
		Goal(
			drone_component = 'Drone',
			goal_component = 'Goal',
			include_z = True if vert_motion else False, # includes z in distance calculations
			tolerance = 0 if airsim_release == 'Tello' else 4,
			name = 'GoalReward',
		)
		rewards.append('GoalReward')
		from rewards.maxsteps import MaxSteps
		MaxSteps(
			name = 'MaxStepsReward',
			max_steps = 4**(1+vert_motion), # base number of steps, will scale with further goal
		)
		rewards.append('MaxStepsReward')
		# REWARDER
		from rewarders.schema import Schema
		Schema(
			rewards_components = rewards,
			reward_weights = reward_weights.copy(),
			name = 'Rewarder',
		)

		# ACTIONS
		actions = []
		if include_resolution:
			from actions.resolution import Resolution 
			Resolution(
				scales_components = [
					'ResizeFlat',
				],
				name = 'FlattenedDepthResolution',
			)
			actions.append('FlattenedDepthResolution')
			if include_bottom:
				Resolution(
					scales_components = [
						'ResizeFlat2',
					],
					name = 'FlattenedDepthResolution2',
				)
				actions.append('FlattenedDepthResolution2')
		if rl_model in ['TD3']:
			base_distance = 10 # meters, will multiply rl_output by this value
			base_yaw = math.pi # degrees, will multiply rl_output by this value
			from actions.move import Move 
			Move(
				drone_component = 'Drone', 
				base_x_rel = base_distance, 
				adjust_for_yaw = adjust_for_yaw,
				name = 'MoveForward',
			)
			actions.append('MoveForward')
			from actions.rotate import Rotate 
			Rotate(
				drone_component = 'Drone',  
				base_yaw = base_yaw,
				min_space = -1, # allows left and right rotations
				name = 'Rotate',
			)
			actions.append('Rotate')
			if vert_motion:
				Move(
					drone_component = 'Drone', 
					base_z_rel = base_distance, 
					min_space = -1, # allows up and down movements
					adjust_for_yaw = adjust_for_yaw,
					name = 'MoveVertical',
				)
				actions.append('MoveVertical')
		if rl_model in ['DQN']:
			from actions.fixedmove import FixedMove 
			FixedMove(
				drone_component = 'Drone', 
				x_speed = 1, 
				adjust_for_yaw = adjust_for_yaw,
				name = 'FixedForward1',
			)
			FixedMove(
				drone_component = 'Drone', 
				x_speed = 5, 
				adjust_for_yaw = adjust_for_yaw,
				name = 'FixedForward2',
			)
			FixedMove(
				drone_component = 'Drone', 
				x_speed = 10, 
				adjust_for_yaw = adjust_for_yaw,
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
			actions.append('FixedForward1')
			actions.append('FixedForward2')
			actions.append('FixedForward3')
			actions.append('FixedRotate1')
			actions.append('FixedRotate2')
			actions.append('FixedRotate3')
			actions.append('FixedRotate4')
			actions.append('FixedRotate5')
			actions.append('FixedRotate6')
			if vert_motion:
				FixedMove(
					drone_component = 'Drone', 
					z_speed = -1, 
					adjust_for_yaw = adjust_for_yaw,
					name = 'FixedUp1',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = -5, 
					adjust_for_yaw = adjust_for_yaw,
					name = 'FixedUp2',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = -10, 
					adjust_for_yaw = adjust_for_yaw,
					name = 'FixedUp3',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = 1, 
					adjust_for_yaw = adjust_for_yaw,
					name = 'FixedDown1',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = 5, 
					adjust_for_yaw = adjust_for_yaw,
					name = 'FixedDown2',
				)
				FixedMove(
					drone_component = 'Drone', 
					z_speed = 10,
					adjust_for_yaw = adjust_for_yaw,
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
			deviation = 0, # start at 0 radians in noise
			deviation_amp = math.radians(1), # amp up noise by 1 degree
			name = 'OrientationNoise',
		)
		GaussianNoise(
			deviation = 0, # start at 0  meters in noise
			deviation_amp = 0.1, # amp up noise by 0.1 meters
			name = 'DistanceNoise',
		)
		from transformers.gaussianblur import GaussianBlur
		GaussianBlur(
			sigma = 0, # start at 0 noise
			sigma_amp = 0.1, # amp up noise by .1 sigma
			name = 'DepthNoise',
		)
		from transformers.normalize import Normalize
		Normalize(
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
			min_input = max_distance / 1000, # min depth (below this is erroneous)
			max_input = max_distance, # max depth
			left = 0, # all values below min_input are erroneous
			name = 'NormalizeDistance',
		)
		Normalize(
			max_input = 255, # MonoDepth2 outputs pixels from 0 to 255
			name = 'NormalizeMD2',
		)
		from transformers.resizeimage import ResizeImage
		image_shape=(84,84) 
		if flat == 'big':
			image_shape=(64,64) 
		if flat == 'big2':
			image_shape=(81,81) 
		ResizeImage(
			image_shape=image_shape,
			name = 'ResizeImage',
		)
		if airsim_release == 'Tello':
			from transformers.monodepth2 import MonoDepth2
			MonoDepth2(
				name = 'MonoDepth2'
			)
		# SENSORS
		# keep track of recent past actions
		from sensors.actions import Actions
		Actions(
			actor_component = 'Actor',
			name = 'ActionsSensor',
			)
		# sense linear distance to goal
		from sensors.distance import Distance
		Distance(
			misc_component = 'Drone',
			misc2_component = 'Goal',
			include_z = False,
			prefix = 'drone_to_goal',
			transformers_components = [
				'DistanceNoise',
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
				'OrientationNoise',
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
					'DistanceNoise',
					'NormalizeDistance',
					],
				name = 'GoalAltitude',
			)
		from sensors.airsimcamera import AirSimCamera
		if policy == 'MlpPolicy':
			# get flattened depth map (obsfucated front facing distance sensors)
			from transformers.resizeflat import ResizeFlat
			max_cols = [8*(i+1) for i in range(8)] # splits depth map by columns
			max_rows = [8*(i+1) for i in range(8)] # splits depth map by rows
			if flat == 'big2':
				max_cols = [9*(i+1) for i in range(9)] # splits depth map by columns
				max_rows = [9*(i+1) for i in range(9)] # splits depth map by rows
			if flat == 'small':
				max_cols = [16, 32, 52, 68, 84] # splits depth map by columns
				max_rows = [21, 42, 63, 84] if vert_motion else [42] # splits depth map by rows
			ResizeFlat(
				max_cols = max_cols,
				max_rows = max_rows,
				name = 'ResizeFlat',
			)
			ResizeFlat(
				max_cols = max_cols,
				max_rows = max_rows,
				name = 'ResizeFlat2',
			)
			if airsim_release == 'Tello':
				from sensors.portcamera import PortCamera
				PortCamera(
					transformers_components = [
						'MonoDepth2',
						'ResizeImage',
						'ResizeFlat',
						'NormalizeMD2',
						],
					name = 'FlattenedDepth',
				)
			else:
				AirSimCamera(
					airsim_component = 'Map',
					transformers_components = [
						'ResizeImage',
						#'DepthNoise',
						'ResizeFlat',
						'DistanceNoise',
						'NormalizeDistance',
						],
					name = 'FlattenedDepth',
					)
				if include_bottom:
					AirSimCamera(
						airsim_component = 'Map',
						camera_view='3', 
						transformers_components = [
							'ResizeImage',
							#'DepthNoise',
							'ResizeFlat2',
							'DistanceNoise',
							'NormalizeDistance',
							],
						name = 'FlattenedDepth2',
						)

		# OBSERVER
		# currently must count vector size of sensor output (TODO: automate this)
		vector_sensors = ['ActionsSensor', 'GoalDistance', 'GoalOrientation']
		vector_length = 1 + 1 # 1 for GoalDistance, 1 for GoalOrientation
		if rl_model in ['DQN']:
			vector_length += 1 # DQN adds only one action for ActionSensor
		if rl_model in ['TD3']:
			vector_length += len(actions) # TD3 adds multiple actions for ActionSensor
		if vert_motion:
			vector_sensors.append('GoalAltitude')
			vector_length += 1 # 1 for GoalAltitude
		if policy == 'MlpPolicy':
			vector_sensors.append('FlattenedDepth')
			vector_length += len(max_cols) * len(max_rows) # several more vector elements
			if include_bottom:
				vector_sensors.append('FlattenedDepth2')
				vector_length += len(max_cols) * len(max_rows) # several more vector elements
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
		if hyper:
			_space = {}
			if 'learning_rate' in hyper_params:
				_space['learning_rate'] = hp.quniform('learning_rate', 1, 8, 1)
			if 'learning_starts'in hyper_params:
				_space['learning_starts'] = hp.quniform('learning_starts', 100, 5_000, 100)
			if 'tau'in hyper_params:
				_space['tau'] = hp.uniform('tau', 0, 1)
			if 'buffer_size'in hyper_params:
				_space['buffer_size'] = hp.quniform('buffer_size', 1_000, replay_buffer_size, 1_000)
			if 'gamma' in hyper_params:
				_space['gamma'] = hp.quniform('gamma', 1, 8, 1)
			if 'batch_size'in hyper_params:
				_space['batch_size'] = hp.quniform('batch_size', 10, 400, 10)
			if 'train_freq'in hyper_params:
				_space['train_freq'] = hp.quniform('train_freq', 1, 6, 1)
			if 'policy_delay'in hyper_params:
				_space['policy_delay'] = hp.quniform('policy_delay', 1, 8, 1)
			if 'target_policy_noise'in hyper_params:
				_space['target_policy_noise'] = hp.uniform('target_policy_noise', 0, 1)
			if 'target_noise_clip'in hyper_params:
				_space['target_noise_clip'] = hp.uniform('target_noise_clip', 0, 1)
			if 'policy_layers'in hyper_params:
				_space['policy_layers'] = hp.quniform('policy_layers', 1, 4, 1)
			if 'policy_nodes'in hyper_params:
				_space['policy_nodes'] = hp.quniform('policy_nodes', 10, 400, 10)
			from models.hyper import Hyper
			Hyper(
				environment_component = 'TrainEnvironment',
				_space = _space,
				model_type = rl_model,
				default_params= {
					'policy': policy,
					'buffer_size': replay_buffer_size,
					'tensorboard_log': working_directory + 'tensorboard_log/',
				},
				name='Model',
			)
		else:
			if rl_model == 'TD3':
				from models.td3 import TD3
				TD3(
					environment_component = 'TrainEnvironment',
					policy = policy,
					buffer_size = replay_buffer_size,
					learning_starts = learning_starts,
					tensorboard_log = working_directory + 'tensorboard_log/',
					read_model_path = read_model_path,
					read_replay_buffer_path = read_replay_buffer_path,
					#action_noise = 'normal',
					action_noise = None,
					name='Model',
				)
			if rl_model == 'DQN':
				from models.dqn import DQN
				DQN(
					environment_component = 'TrainEnvironment',
					policy = policy,
					buffer_size = replay_buffer_size,
					learning_starts = learning_starts,
					tensorboard_log = working_directory + 'tensorboard_log/',
					read_model_path = read_model_path,
					read_replay_buffer_path = read_replay_buffer_path,
					name='Model',
				)


		# CREATE MODIFIERS
		# SPAWNER
		from modifiers.spawner import Spawner
		from others.spawn import Spawn
		Spawner(
			base_component = 'Drone',
			parent_method = 'reset',
			drone_component = 'Drone',
			spawns_components=[
				Spawn(
					map_component = 'Map',
					bounds_component = 'TrainingBounds',
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
					x=0,
					y=0,
					dz=4,
					yaw=math.radians(0),
					),
				Spawn(
					x=0,
					y=0,
					dz=4,
					yaw=math.radians(45),
					),
				Spawn(
					x=0,
					y=0,
					dz=4,
					yaw=math.radians(135),
					),
				Spawn(
					x=0,
					y=0,
					dz=4,
					yaw=math.radians(180),
					),
				Spawn(
					x=0,
					y=0,
					dz=4,
					yaw=math.radians(-130),
					),
				Spawn(
					x=0,
					y=0,
					dz=4,
					yaw=math.radians(-45),
					),
			],
			order='post',
			on_train = False,
			name='EvaluateSpawner',
		)
		# EVALUATOR
		nEvalEpisodes = 1 if airsim_release == 'Tello' else 6
		from modifiers.evaluatorcharlie import EvaluatorCharlie
		# Evaluate model after each epoch (checkpoint)
		EvaluatorCharlie(
			base_component = 'TrainEnvironment',
			parent_method = 'reset',
			order = 'pre',
			evaluate_environment_component = 'EvaluateEnvironment',
			goal_component = 'Goal',
			model_component = 'Model',
			noises_components = [
				'OrientationNoise', 
				'DistanceNoise', 
				],
			spawn_bounds_component = 'TrainingBounds',
			nEpisodes = nEvalEpisodes,
			frequency = checkpoint,
			track_vars = [],
			save_every_model = True,
			counter = 0, # -1 offset to do an eval before any training
			name = 'Evaluator',
		)
		if not hyper:
			# SAVERS
			from modifiers.saver import Saver
			# save Train states and observations after each epoch (checkpoint)
			Saver(
				base_component = 'TrainEnvironment',
				parent_method = 'end',
				track_vars = [
							'observations', 
							'states',
							],
				order = 'post',
				save_config = True,
				save_benchmarks = True,
				frequency = checkpoint,
				name='TrainEnvSaver',
			)
			# save model after each epoch (checkpoint)
			# environment does not have access to model
			Saver(
				base_component = 'Model',
				parent_method = 'end',
				track_vars = [
							'model', 
							'replay_buffer',
							],
				order = 'post',
				frequency = nEvalEpisodes,
				name='ModelSaver',
			)
			# save Evlaluate states and observations after each epoch (checkpoint)
			Saver(
				base_component = 'EvaluateEnvironment',
				parent_method = 'end',
				track_vars = [
							'observations', 
							'states',
							],
				order = 'post',
				frequency = nEvalEpisodes,
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
		replay_buffer_size = replay_buffer_size, # a size of 1_000_000 requires 56.78 GB if using MultiInputPolicy
		continue_training = continue_training, # set to true if continuing training from checkpoint
		controller_type = controller_type, # Train, Debug, Drift, Evaluate
		actor = actor, # Teleporter Continuous
		clock_speed = clock_speed, # airsim clock speed (increasing this will also decerase sim-quality)
		training_steps = training_steps, # max number of training steps 
		max_distance = max_distance, # distance contraint used for several calculations (see below)
		nTimesteps = nTimesteps, # number of timesteps to use in observation space
		checkpoint = checkpoint, # evaluate model and save checkpoint every # of episodes
		run_post = run_post, # optionally add text to generated run name (such as run2, retry, etc...)
		distance_reward = distance_reward, # reward function that penalizes distance to goal (large positive fore reaching)
		step_reward = step_reward, # reward function that penalizes longer episode length
		flat = flat, # determines size of flattened depth sensor array 
		hyper = hyper, # optional hyper search over specified parameters using a Gaussian process
		hyper_params = hyper_params, # which hyper parameters to hyper tune if model is type hyper
		read_model_path = read_model_path, # load pretrained model?
		read_replay_buffer_path = read_replay_buffer_path, # load prebuilt replay buffer?
		reward_weights = reward_weights, # reward weights in order: goal, collision, steps
		learning_starts = learning_starts, # how many steps to collect in buffer before training starts
		tello_goal = tello_goal,
		adjust_for_yaw = adjust_for_yaw,
		include_resolution = include_resolution,
		include_bottom = include_bottom,
		action_noise = action_noise,
)

# make dir to save all tello imgs to
tell_img_path = utils.get_global_parameter('working_directory') + 'tello_imgs/'
if airsim_release == 'Tello' and not os.path.exists(tell_img_path):
	os.makedirs(tell_img_path)

# create any other components
if not continue_training:
	# add components here
	pass

# run baby run
run_controller(configuration)
