import rl_utils as utils
from configuration import Configuration
import sys
import copy
import torch
import numpy as np
import math
import random

## grab arguments input from terminal
args = sys.argv
# first sys argument is test_case to run (see options below) - usually changes by device
	# if no arguments will default to test_case 'h4' - my work laptop
test_case = ''
if len(args) > 1:
	test_case = args[1].lower()
# second sys argument is to continue training from last checkpoint (True) or not (False)
	# will assume False if no additional input
	# must pass in run_post var
continue_training = False # if True will continue learning loop from last step saved, if False will reset learning loop
if len(args) > 2:
	continue_training = args[2] in ['true', 'True']
# third sys argument is any text to concatenate to run output folder name (i.e. run2 etc)
	# will assume no text to concat if no additional input
run_post = ''
if len(args) > 3:
	run_post = args[3]

## set random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

## set params
repo_version = 'delta1'
airsim_release = 'Blocks'
read_model_path = None
read_replay_buffer_path = None
replay_buffer_size = 400_000 # number of recent samples (steps) to save in replay buffer
	#400_000 will work well within a 32gb-RAM system when using MultiInputPolicy
clock_speed = 10 # airsim clock speed (increasing this will also decerase sim-quality)
distance_param = 125 # distance contraint used for several calculations (see below)
nTimesteps = 4 # number of timesteps to include in observation space
# sensors?
vector_sensors = {
	'FlattenedDepth1',
	#'FlattenedDepth2',
	'GoalDistance',
	'GoalOrientation',
	#'GoalAltitude',
}
# actions?
actions = [
	'MoveForward',
	'Rotate',
	#'MoveVertical',
	#'SlimAction',
	#'FlattenedDepthResolution1',
	#'FlattenedDepthResolution2',
]
# rewards and weights?
reward_norm = 1
rewards = {
	'CollisionReward': 200/reward_norm, 
	'GoalReward': 200/reward_norm,
	'StepsReward': 2/reward_norm,
	'DistanceReward': 0.1/reward_norm,
	#'SlimReward': 3/reward_norm,
	#'ResolutionReward1': 0.5/reward_norm,
	#'ResolutionReward2': 0.5/reward_norm,
	'MaxStepsReward': 0/reward_norm,
}
if test_case in ['h4', 'torch']:
	vert_motion = False
if test_case in ['pyro', 'phoenix']:
	vert_motion = True
use_slim = False
use_res = False
child_project = 'navi'
run_name = child_project + '_' + airsim_release 
run_name += '_vert' if vert_motion else '_horz' 
run_name += '_' + test_case + '_' + repo_version
if run_post != '': 
	run_name += '_' + run_post
# learning loop (controller) stuff
continue_training = False
max_episodes = 100_000 # max number of episodes to train for before terminating learning loop
	# computations will finish roughly 250k steps a day (episode lengths vary but ~10-20 per)
checkpoint = 100 # evaluate model and save checkpoint every # of episodes
train_start = 100 # collect this many episodes before start updating networks
train_freq = 1
num_batches = -1
random_start = 50
batch_size = 100
with_distillation = False
use_wandb = False
parent_project = 'eecs298'
controller_type = 'Train' # Train Debug Drift Evaluate Data	
controller_params = {
	'model_component' : 'Model',
	'train_environment_component' : 'TrainEnvironment',
	'continue_training' : continue_training,
	'max_episodes' : max_episodes,
	'random_start' : random_start, # num episodes to take random actions?
	'train_start' : train_start, # don't call train() until after train_start episodes
	'train_freq' : train_freq, # then call train() every train_freq episode
	'batch_size' : batch_size, # split training into mini-batches of steps from buffer
	'num_batches' : num_batches, # split training into mini-batches of steps from buffer
	'with_distillation' : with_distillation, # slims during train() and distills to output of super
	'use_wandb' : use_wandb, # turns on logging to wandb
	'project_name' : parent_project + '_' + child_project, # wandb logs here
}
if test_case in ['h4', 'pyro']:
	sb3 = False
if test_case in ['torch', 'phoenix']:
	sb3 = True

## runs overarching base code
def create_base_components(
		actions,
		rewards,
		airsim_release = 'Blocks', # name of airsim release to use, see maps.arisimmap
		vert_motion = False, # allowed to move on z-axis? False will restrict motion to horizontal plane
		replay_buffer_size = 1_000_000, # a size of 1_000_000 requires 56.78 GB if using MultiInputPolicy
		continue_training = False, # set to true if continuing training from checkpoint
		controller_type = 'Train', # Train, Debug, Drift, Evaluate
		controller_params = {},
		clock_speed = 10, # airsim clock speed (increasing this will alsovalue_typevalue_type
		distance_param = 125,
		nTimesteps = 4,
		checkpoint = 100, # evaluate model and save checkpoint every # of episodes
		read_model_path = None, # load pretrained model?
		read_replay_buffer_path = None, # load prebuilt replay buffer?
		use_slim = False,
		use_res = False,
		run_name = 'run',
		repo_version = 'unknown',
		sb3 = False,
):

	# **** SETUP ****
	OS = utils.setup(
		write_parent = 'local/runs/',
		run_prefix = run_name,
		)
	working_directory = utils.get_global_parameter('working_directory')


	## CONTROLLER
	controller = utils.get_controller(
		controller_type = controller_type,
		controller_params = controller_params, 
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


	# READ CONFIGURATION?
	read_configuration_path = working_directory + 'configuration.json'
	update_meta = True
	if continue_training:
		# load configuration file and create object to save and connect components
		configuration = Configuration.load(read_configuration_path, controller)
		if update_meta:
			configuration.update_meta(meta)
		# load model weights and replay buffer
		# YOU MAY HAVE TO ADJUST THIS FOR YOUR USE
		read_model_path = working_directory + 'Model/model.zip'
		read_replay_buffer_path = working_directory + 'Model/replay_buffer.zip'
		_model = configuration.get_component('Model')
		_model.model = read_model_path
		_model.replay_buffer = read_replay_buffer_path


	# NEW CONFIGURATION?
	else:

		## CONFIGURATION 
		configuration = Configuration(
			meta, 
			controller, 
			add_timers=False, # auto times all function calls
			add_memories=False, # logs memory of all objs at each checkpoint
			)


		# **** CREATE COMPONENTS ****
		

		## MAP
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
			#voxels_component='Voxels',
			release_path = release_path,
			settings = {
				'ClockSpeed': clock_speed,
				},
			setting_files = [
				'lightweight', # see maps/airsim_settings/...
				],
			console_flags = console_flags.copy(),
			name = 'Map',
		)
		# voxels grabs locations of objects from airsim map
		# used to validate spawn and goal points (not inside an object)
		# also used to visualize flight paths
		""" from others.voxels import Voxels
		Voxels(
			relative_path = working_directory + 'map_voxels.binvox',
			map_component = 'Map',
			x_length = 2 * distance_param, # total x-axis meters (split around center)
			y_length = 2 * distance_param, # total y-axis  meters (split around center)
			z_length = 2 * distance_param, # total z-axis  meters (split around center)
			name = 'Voxels',
		) """


		#  DRONE
		from drones.airsimdrone import AirSimDrone
		AirSimDrone(
			airsim_component = 'Map',
			name='Drone',
		)

		## ACTIONS		
		base_distance = 10 # meters, will multiply rl_output by this value
		base_yaw = math.pi # degrees, will multiply rl_output by this value
		if 'MoveForward' in actions:
			from actions.move import Move 
			Move(
				drone_component = 'Drone', 
				base_x_rel = base_distance, 
				adjust_for_yaw = True,
				zero_thresh_abs = False, # any negative input is not move forward
				name = 'MoveForward',
			)
		if 'Rotate' in actions:
			from actions.rotate import Rotate 
			Rotate(
				drone_component = 'Drone',  
				base_yaw = base_yaw,
				name = 'Rotate',
			)
		if 'MoveVertical' in actions:
			from actions.move import Move 
			Move(
				drone_component = 'Drone', 
				base_z_rel = base_distance, 
				adjust_for_yaw = True,
				active = vert_motion,
				name = 'MoveVertical',
			)
		if 'SlimAction' in actions:
			from actions.slim import Slim
			Slim(
				model_component = 'Model',
				active = use_slim,
				name = 'SlimAction'
			) 
		if 'FlattenedDepthResolution1' in actions:
			from actions.resolution import Resolution 
			Resolution(
				scales_components = [
					'ResizeFlat1',
				],
				active = use_res,
				name = 'FlattenedDepthResolution1',
			)
		if 'FlattenedDepthResolution2' in actions:
			from actions.resolution import Resolution 
			Resolution(
				scales_components = [
					'ResizeFlat2',
				],
				active = use_res,
				name = 'FlattenedDepthResolution2',
			)
		# ACTOR (teleporter is more stable, and quicker)
		from actors.teleporter import Teleporter
		Teleporter(
			drone_component = 'Drone',
			actions_components = actions,
			name='Actor',
		)


		## REWARDS
		# heavy penalty for collision
		if 'CollisionReward' in rewards:
			from rewards.collision import Collision
			Collision(
				drone_component = 'Drone',
				name = 'CollisionReward',
			)
		# increasing reward as approaches goal
		if 'GoalReward' in rewards:
			from rewards.goal import Goal
			Goal(
				drone_component = 'Drone',
				goal_component = 'Goal',
				include_z = True, # includes z in distance calculations
				tolerance = 2,
				name = 'GoalReward',
			)
		# penalize heavier as approaches time constraint
		if 'StepsReward' in rewards:
			from rewards.steps import Steps
			Steps(
				name = 'StepsReward',
			)
		# increasing reward as approaches goal
		if 'DistanceReward' in rewards:
			from rewards.distance import Distance
			Distance(
				drone_component = 'Drone',
				goal_component = 'Goal',
				include_z = True, # includes z in distance calculations
				name = 'DistanceReward',
			)
		# penalize computational complexity
		if 'SlimReward' in rewards:
			from rewards.slim import Slim
			Slim(
				slim_component='SlimAction',
				name='SlimReward',
			)
		# penalty for higher resolutions=
		if 'ResolutionReward1' in rewards:
			from rewards.resolution import Resolution
			Resolution(
				resolution_component = 'FlattenedDepthResolution1',
				name = 'ResolutionReward1',
			)
		if 'ResolutionReward2' in rewards:
			from rewards.resolution import Resolution
			Resolution(
				resolution_component = 'FlattenedDepthResolution2',
				name = 'ResolutionReward2',
			)
		# do not exceed this many steps
		if 'MaxStepsReward' in rewards:
			from rewards.maxsteps import MaxSteps
			MaxSteps(
				name = 'MaxStepsReward',
				max_steps = 30, # base number of steps, will scale with further goal
			)
		# REWARDER
		from rewarders.schema import Schema
		Schema(
			rewards_components = list(rewards.keys()),
			reward_weights = [rewards[key] for key in rewards],
			name = 'Rewarder',
		)


		## OBSERVATION SPACE
		# TRANSFORMERS
		#distance_epsilon = distance_param/1000 # outputs a senosr-value of zero for values below this
		from transformers.normalize import Normalize
		Normalize(
			max_input = 2*math.pi, # max angle
			name = 'NormalizeOrientation',
		)
		Normalize(
			#min_input = distance_epsilon, # min depth (below this is erroneous)
			max_input = distance_param, # max depth
			#left = 0, # set all values below range to this
			name = 'NormalizeDistance',
		)
		from transformers.resizeimage import ResizeImage
		image_shape=(25,25)
		ResizeImage(
			image_shape=image_shape,
			name = 'ResizeImage',
		)
		# SENSORS
		vector_length = 0
		# sense horz distance to goal
		if 'GoalDistance' in vector_sensors:
			from sensors.distance import Distance
			Distance(
				misc_component = 'Drone',
				misc2_component = 'Goal',
				include_z = False,
				prefix = 'drone_to_goal',
				transformers_components = [
					'NormalizeDistance',
					], 
				name = 'GoalDistance',
			)
			vector_length += 1
		# sense yaw difference to goal 
		if 'GoalOrientation' in vector_sensors:
			from sensors.orientation import Orientation
			Orientation(
				misc_component = 'Drone',
				misc2_component = 'Goal',
				prefix = 'drone_to_goal',
				transformers_components = [
					'NormalizeOrientation',
					],
				name = 'GoalOrientation',
			)
			vector_length += 1
		# sense vert distance to goal
		if 'GoalAltitude' in vector_sensors:
			Distance(
				misc_component = 'Drone',
				misc2_component = 'Goal',
				include_x = False,
				include_y = False,
				prefix = 'drone_to_goal',
				transformers_components = [
					'NormalizeDistance',
					],
				name = 'GoalAltitude',
			)
			vector_length += 1
		# get flattened depth map (obsfucated front facing distance sensors)
		max_cols = [5*(i+1) for i in range(5)] # splits depth map by columns
		max_rows = [5*(i+1) for i in range(5)] # splits depth map by rows
		if 'FlattenedDepth1' in vector_sensors:
			from transformers.resizeflat import ResizeFlat
			ResizeFlat(
				max_cols = max_cols,
				max_rows = max_rows,
				name = 'ResizeFlat1',
			)
			# forward facing depth map
			from sensors.airsimcamera import AirSimCamera
			AirSimCamera(
				airsim_component = 'Map',
				transformers_components = [
					'ResizeImage',
					'ResizeFlat1',
					'NormalizeDistance',
					],
				name = 'FlattenedDepth1',
			)
			vector_length += len(max_cols) * len(max_rows)
		if 'FlattenedDepth2' in vector_sensors:
			from transformers.resizeflat import ResizeFlat
			ResizeFlat(
				max_cols = max_cols,
				max_rows = max_rows,
				name = 'ResizeFlat2',
			)
			# downward facing depth map
			from sensors.airsimcamera import AirSimCamera
			AirSimCamera(
				airsim_component = 'Map',
				camera_view='3', 
				transformers_components = [
					'ResizeImage',
					'ResizeFlat2',
					'NormalizeDistance',
					],
				name = 'FlattenedDepth2',
			)
			vector_length += len(max_cols) * len(max_rows)
		# OBSERVER
		from observers.single import Single
		Single(
			sensors_components = vector_sensors, 
			vector_length = vector_length,
			nTimesteps = nTimesteps,
			name = 'Observer',
		)


		## MODEL
		# make neural networks
		if not sb3:
			#torch.set_default_dtype(torch.float64)
			def create_sequential(
				input_dim,
				output_dim,
				nLayers = 3,
				nNodes = 2**5,
				activation_fn = torch.nn.ReLU,
				with_bias = True,
			):
				net_arch = [nNodes for _ in range(nLayers)]
				modules = []
				if nLayers == 0:
					modules.append(torch.nn.Linear(input_dim, output_dim, bias=with_bias))
				else:
					modules.append(torch.nn.Linear(input_dim, net_arch[0], bias=with_bias))
					modules.append(activation_fn())
					for idx in range(len(net_arch) - 1):
						modules.append(torch.nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
						modules.append(activation_fn())
					modules.append(torch.nn.Linear(net_arch[-1], output_dim, bias=with_bias))
				modules.append(torch.nn.Tanh())
				network = torch.nn.Sequential(*modules)
				return network
			def attach_optimizer(
				network,
				learning_rate = 10**(int(-1*3)),
				weight_decay = 10**(int(-1*6)),
			):
				network.optimizer = torch.optim.Adam(
					network.parameters(),
					lr=learning_rate,
					weight_decay= weight_decay,
				)
			# actor
			input_dim_actor = vector_length * nTimesteps
			output_dim_actor = len(actions)
			actor = create_sequential(input_dim_actor, output_dim_actor)
			attach_optimizer(actor)
			actor_target = create_sequential(input_dim_actor, output_dim_actor)
			actor_target.load_state_dict(copy.deepcopy(actor.state_dict()))
			attach_optimizer(actor_target)
			# critic
			input_dim_critic = input_dim_actor + output_dim_actor
			output_dim_critic = 1
			nCritics = 2
			critics = []
			critics_target = []
			for c in range(nCritics):
				critic = create_sequential(input_dim_critic, output_dim_critic)
				critics.append(critic)
				attach_optimizer(critic)
				critic_target = create_sequential(input_dim_critic, output_dim_critic)
				critic_target.load_state_dict(copy.deepcopy(critic.state_dict()))
				critics_target.append(critic_target)
				attach_optimizer(critic_target)
		# TD3
		from models.td3 import TD3
		TD3(
			actor=None if sb3 else actor,
			actor_target=None if sb3 else actor_target,
			critics=None if sb3 else critics,
			critics_target=None if sb3 else critics_target,
			obs_shape=[input_dim_actor],
			act_shape=[output_dim_actor],
			write_dir=working_directory+'Model/',
			buffer_size=replay_buffer_size,
			sb3=sb3,
			name='Model',
		)

		# spawn (initial drone pos and goal pos)
		from others.spawn import Spawn
		motion = 'horizontal'
		if vert_motion:
			motion = 'vertical'
		Spawn(
			#read_path='aPaths_' + motion + '_train.p', # read in dict of possible paths or static spawns
			#random=True, # True will get random path, False will use static
			#name='SpawnTrain'
			read_path='aPaths_' + motion + '_val.p', # read in dict of possible paths or static spawns
			random=True, # True will get random path, False will use static
			name='SpawnTrain'
		)
		nEvals = 100
		Spawn(
			read_path='spawns_' + motion + '_val.p', # read in dict of possible paths or static spawns
			random=False, # True will get random path, False will use static
			clip_spawns=nEvals,
			name='SpawnEvaluate'
		)

		## TRAIN ENVIRONMENT
		from environments.goalenv import GoalEnv
		GoalEnv(
			drone_component='Drone', 
			actor_component='Actor', 
			observer_component='Observer', 
			rewarder_component='Rewarder', 
			spawn_component='SpawnTrain',
			goal_component='Goal',
			model_component='Model',
			map_component='Map',
			evaluator_component='Evaluator',
			name='TrainEnvironment',
		)
		## EVALUATE ENVIRONMENT
		GoalEnv(
			drone_component='Drone', 
			actor_component='Actor', 
			observer_component='Observer', 
			rewarder_component='Rewarder', 
			spawn_component='SpawnEvaluate',
			goal_component='Goal',
			model_component='Model',
			map_component='Map',
			name='EvaluateEnvironment',
		)

		# goals
		from others.goal import Goal
		Goal(
			name = 'Goal',
		)

		# EVALUATOR
		# curriculum learning
		from others.fvar import FVar
		FVar(
			fpath = 'temp/evaluator_fvar.p',
			default = [0],
			name = 'EvaluatorFVar',
		)
		FVar(
			fpath = 'temp/trainer_fvar.p',
			default = [0],
			name = 'TrainerFVar',
		)
		from evaluators.curriculum import Curriculum
		Curriculum(
			models_directory=working_directory+'Model/',
			model_component='Model',
			eval_freq=checkpoint,
			is_evaluator=True,
			evaluator_fvar_component='EvaluatorFVar',
			evaluate_environment_component='EvaluateEnvironment',
			nEpisodes=nEvals,
			criteria=99,
			is_trainer=True,
			trainer_fvar_component='TrainerFVar',
			train_spawn_component='SpawnTrain',
			steps=list(range(1,21)),
			in_final_form=False,
			level=0,
			level_steps=1,
			name = 'Evaluator',
		)

		# SAVERS - writes values to file
		from modifiers.saver import Saver
		# save Train states and observations after each epoch (checkpoint)
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
			name='TrainEnvSaver',
		)
		# save Evlaluate states and observations after each epoch (checkpoint)
		Saver(
			base_component = 'EvaluateEnvironment',
			parent_method = 'end',
			track_vars = [
						'observations', 
						'states',
						],
			order = 'pre',
			frequency = nEvals,
			name='EvalEnvSaver',
		)
	
	return configuration


def run_controller(configuration):
	utils.speak('configuration created!')

	# CONNECT COMPONENTS
	configuration.connect_all()
	
	utils.speak('all components connected.')

	# WRITE CONFIGURATION
	configuration.save()

	# RUN CONTROLLER
	configuration.controller.run()

	# done
	configuration.controller.stop()

## create base components
configuration = create_base_components(
		actions,
		rewards,
		airsim_release = airsim_release, # name of airsim release to use, see maps.arisimmap
		vert_motion = vert_motion, # allowed to move on z-axis? False will restrict motion to horizontal plane
		replay_buffer_size = replay_buffer_size, # a size of 1_000_000 requires 56.78 GB if using MultiInputPolicy
		continue_training = continue_training, # set to true if continuing training from checkpoint
		controller_type = controller_type, # Train, Debug, Drift, Evaluate
		controller_params = controller_params,
		clock_speed = clock_speed, # airsim clock speed (increasing this will also decerase sim-quality)
		distance_param = distance_param, # distance contraint used for several calculations (see below)
		nTimesteps = nTimesteps, # number of timesteps to use in observation space
		checkpoint = checkpoint, # evaluate model and save checkpoint every # of episodes
		read_model_path = read_model_path, # load pretrained model?
		read_replay_buffer_path = read_replay_buffer_path, # load prebuilt replay buffer?
		use_slim = use_slim,
		use_res = use_res,
		run_name = run_name,
		repo_version = repo_version,
		sb3 = sb3,
)

## create any other components
if not continue_training:
	# ---* add components here *--- #
	pass

## run baby run
run_controller(configuration)