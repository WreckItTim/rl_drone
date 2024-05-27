import rl_utils as utils
from configuration import Configuration
import math
import numpy as np
import sys
import os

## PARAMS
run_post = ''

# local params
utils.read_global_parameters()
utils.set_operating_system()
instance_name = utils.get_global_parameter('instance_name')
device_name = 'cuda:0'
if instance_name in ['mlserver2019L']:
    device_name = 'cuda:2'
render_screen = utils.get_global_parameter('render_screen')
console_flags = ['-Windowed']
if not render_screen:
	console_flags = ['-RenderOffscreen']
OS = utils.get_global_parameter('OS')
	
# experiment params
cnn_model_name = 'GT'
airsim_map = 'Blocks'
release_path = 'local/airsim_maps/' + airsim_map + '/' + OS + 'NoEditor/' + airsim_map
if OS in ['Linux']:
	release_path += '.sh'
if OS in ['Windows']:
	release_path += '.exe'
run_post += cnn_model_name + '_' + airsim_map + '_' + instance_name
vertical = True
dz = 4

# **** SETUP ****
run_name = 'navi'
run_name += '_vert' if vertical else '_horz'
run_name += '_' + run_post
continue_training = False # os.path.exists('local/runs/' + run_name)
utils.setup(
	write_parent = 'local/runs/',
	run_name = run_name,
	)
working_directory = utils.get_global_parameter('working_directory')

## CONTROLLER
controller = utils.get_controller(
	controller_type = 'Train',
	total_timesteps = 1_000_000, # optional if using train - all other hypers set from model instance
	continue_training = continue_training, # if True will continue learning loop from last step saved, if False will reset learning loop
	model_component = 'Model', # if using train, set model
	environment_component = 'Environment', # if using train, set train environment
	log_interval = 999999,
	evaluator_component = 'Evaluator',
	project_name = 'pretest',
)
# SET META DATA (anything you want here, just writes to config file as a dict)
meta = {
	'author_info': 'Timothy K Johnsen, tim.k.johnsen@gmail.com',
	'repo_version': 'navislim',
	'run_name': utils.get_global_parameter('run_name'),
	'timestamp': utils.get_timestamp(),
	'run_OS': utils.get_global_parameter('OS'),
	'absolute_path' : utils.get_global_parameter('absolute_path'),
	'working_directory' : working_directory,
	'continued_from' : [],
	}

utils.speak('creating ' + run_name + ' ...')

# READ CONFIGURATION?
read_configuration_path = working_directory + 'configuration.json'
update_meta = True
if continue_training:
	# load configuration file and create object to save and connect components
	configuration = Configuration.load(read_configuration_path, controller)
	if update_meta:
		if 'continued_from' in   configuration.meta:      
			meta['continued_from'] = configuration.meta['continued_from'].copy() + [configuration.meta['timestamp']]
		else:            
			meta['continued_from'] = [configuration.meta['timestamp']]
		configuration.update_meta(meta)
	# load model weights and replay buffer
	read_model_path = working_directory + 'Model/model.zip'
	read_replay_buffer_path = working_directory + 'Model/replay_buffer.zip'
	_model = configuration.get_component('Model')
	_model.read_model_path = read_model_path
	_model.read_replay_buffer_path = read_replay_buffer_path
	print('read configuration from continue training')

else:
	## CONFIGURATION 
	configuration = Configuration(
		meta, 
		controller, 
		add_timers=False, 
		add_memories=False,
		)


	# **** CREATE COMPONENTS ****


	## TRAIN ENVIRONMENT
	from environments.goalenv import GoalEnv
	GoalEnv(
		drone_component='Drone', 
		actor_component='Actor', 
		observer_component='Observer', 
		rewarder_component='Rewarder', 
		goal_component='Goal',
		map_component='Map',
		model_component='Model',
		others_components=['Evaluator'],
		name='Environment',
	)

	## MAP
	from maps.airsimmap import AirSimMap
	# create airsim map object
	AirSimMap(
		voxels_component='Voxels',
		release_path = release_path,  
		settings = {
			'ClockSpeed': 8,
			},
		setting_files = [
			'lightweight', # see maps/airsim_settings
			],
		console_flags = console_flags,
		name = 'Map',
	)
	# voxels grabs locations of objects from airsim map
	# used to validate spawn and goal points (not inside an object)
	# also used to visualize flight paths
	from others.voxels import Voxels
	Voxels(
		relative_path = working_directory + 'map_voxels.binvox',
		map_component = 'Map',
		x_length = 2 * 125, # total x-axis meters (split around center)
		y_length = 2 * 125, # total y-axis  meters (split around center)
		z_length = 2 * 125, # total z-axis  meters (split around center)
		name = 'Voxels',
		)

	# MAP BOUNDS
	from others.boundscube import BoundsCube
	BoundsCube(
		center = [0, 0, 0],
		x = [-125, 125],
		y = [-125, 125],
		z = [-40, -1],
		name = 'MapBounds'
		)


	#  DRONE
	from drones.airsimdrone import AirSimDrone
	AirSimDrone(
		airsim_component = 'Map',
		name='Drone',
	)

		
	## GOAL
	# dynamic goal will spawn in bounds - randomly for train, static for evaluate
	# goal distance will increase, "amp up", with curriculum learning
	from others.relativegoal import RelativeGoal
	RelativeGoal(
		drone_component = 'Drone',
		map_component = 'Map',
		bounds_component = 'MapBounds',
		static_r = 5,
		static_dz = dz,
		random_r = [5,10], # relative distance for random goal from drone
		random_dz = [dz,dz], # relative z for random goal from drone (this is dz above roof or floor)
		random_yaw = [-1*np.pi, np.pi], # relative yaw for random goal from drone
		vertical = vertical,
		random_point = True,
		name = 'Goal',
	)


	## REWARDS
	rewards = []
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
		include_z = vertical, # includes z in distance calculations
		tolerance = 4,
		name = 'GoalReward',
	)
	rewards.append('GoalReward')
	# penalize heavier as approaches time constraint
	from rewards.steps import Steps
	Steps(
		name = 'StepsReward',
	)
	rewards.append('StepsReward')
	# increasing reward as approaches goal
	from rewards.distance import Distance
	Distance(
		drone_component = 'Drone',
		goal_component = 'Goal',
		include_z = vertical, # includes z in distance calculations
		name = 'DistanceReward',
	)
	rewards.append('DistanceReward')
	# do not exceed this many steps
	from rewards.maxsteps import MaxSteps
	MaxSteps(
		update_steps = True,
		max_steps = 4, # base number of steps, will scale with further goal
		max_max = 50,
		name = 'MaxStepsReward',
	)
	rewards.append('MaxStepsReward')
	# REWARDER
	from rewarders.schema import Schema
	Schema(
		rewards_components = rewards,
		reward_weights = [1, 1, 1/10, 1/100, 0],
		name = 'Rewarder',
	)


	## ACTIONS
	actions = []			
	base_distance = 10 # meters, will multiply rl_output by this value
	base_yaw = math.pi # degrees, will multiply rl_output by this value
		
	from actions.move import Move 
	Move(
		drone_component = 'Drone', 
		base_x_rel = base_distance, 
		adjust_for_yaw = True,
		zero_thresh_abs = False, # any negative input is not move forward
		name = 'MoveForward',
	)
	actions.append('MoveForward')
		
	if vertical:
		from actions.move import Move 
		Move(
			drone_component = 'Drone', 
			base_z_rel = base_distance,
			name = 'MoveVertical',
		)
		actions.append('MoveVertical')
		
	from actions.rotate import Rotate 
	Rotate(
		drone_component = 'Drone',  
		base_yaw = base_yaw,
		name = 'Rotate',
	)
	actions.append('Rotate')


	## ACTOR
	from actors.teleporter import Teleporter
	Teleporter(
		drone_component = 'Drone',
		actions_components = actions,
		name='Actor',
	)


	## OBSERVATION SPACE
	# TRANSFORMERS
	from transformers.normalize import Normalize
	Normalize(
		max_input = 2*math.pi, # max angle
		name = 'NormalizeOrientation',
	)
	Normalize(
		min_input = -100, # min depth
		max_input = 100, # max depth
		name = 'NormalizeDistance',
	)
	if cnn_model_name not in ['GT']:
		from transformers.cnn import CNN
		CNN(
			model_path = cnn_model_name + '.pt', # will create models.model_name() load weights from local/models/model_name.pt
			device_name = device_name, # will load model on this device (cpu or cuda)
			name = 'RGB2Depth',
		)
	# from transformers.resizeimage import ResizeImage
	# image_shape=(25,25)
	# ResizeImage(
	# 	image_shape=image_shape,
	# 	name = 'ResizeImage',
	# )
	# SENSORS
	# sense horz distance to goal
	from sensors.distance import Distance
	Distance(
		misc_component = 'Drone',
		misc2_component = 'Goal',
		include_z = False,
		prefix = 'drone_to_goal_xy',
		transformers_components = [
			'NormalizeDistance',
			], 
		name = 'GoalDistanceXY',
	)
	if vertical:
		Distance(
			misc_component = 'Drone',
			misc2_component = 'Goal',
			include_x = False,
			include_y = False,
			add_in_quad = False,
			prefix = 'drone_to_goal_z',
			transformers_components = [
				'NormalizeDistance',
				], 
			name = 'GoalDistanceZ',
		)
	# sense yaw difference to goal 
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
	# get flattened depth map (obsfucated front facing distance sensors)
	from transformers.resizeflat import ResizeFlat
	max_rows = [24*(i+1) for i in range(6)] # splits depth map by columns
	max_cols = [32*(i+1) for i in range(8)] # splits depth map by rows
	ResizeFlat(
		max_cols = max_cols,
		max_rows = max_rows,
		name = 'ResizeFlat',
	)
	from sensors.airsimcamera import AirSimCamera
	if cnn_model_name not in ['GT']:
		AirSimCamera(
			airsim_component = 'Map',
			image_type = 0, 
			is_gray = False,
			as_float = False,
			transformers_components = [
				'RGB2Depth',
				'ResizeFlat',
				],
			name = 'FlattenedDepth',
		)
	else:
		AirSimCamera(
			airsim_component = 'Map',
			transformers_components = [
				'ResizeFlat',
				'NormalizeDistance',
				],
			name = 'FlattenedDepth',
		)
	if vertical:
		max_rows2 = [48*(i+1) for i in range(3)] # splits depth map by columns
		max_cols2 = [64*(i+1) for i in range(4)] # splits depth map by rows
		ResizeFlat(
			max_cols = max_cols2,
			max_rows = max_rows2,
			name = 'ResizeFlat2',
		)
		AirSimCamera(
			airsim_component = 'Map',
			camera_view = '3',
			transformers_components = [
				'ResizeFlat2',
				'NormalizeDistance',
				],
			name = 'BellySensor',
		)
	# OBSERVER
	# currently must count vector size of sensor output (TODO: automate this)
	vector_sensors = []
	vector_length = 0
	vector_sensors.append('FlattenedDepth')
	vector_length += len(max_cols) * len(max_rows) # several more vector elements
	vector_sensors.append('GoalDistanceXY')
	vector_length += 1
	vector_sensors.append('GoalOrientation')
	vector_length += 1
	if vertical:
		vector_sensors.append('GoalDistanceZ')
		vector_length += 1
		vector_sensors.append('BellySensor')
		vector_length += len(max_cols2) * len(max_rows2) # several more vector elements
	from observers.single import Single
	Single(
		sensors_components = vector_sensors, 
		vector_length = vector_length,
		nTimesteps = 4,
		name = 'Observer',
	)


	## MODEL
	from drls.td3 import TD3
	TD3(
		environment_component = 'Environment',
		policy = 'MlpPolicy',
		policy_kwargs = {'net_arch':[64,32,32]},
		buffer_size = 400_000,
		learning_starts = 100,
		train_freq = (1, "episode"),
		tensorboard_log = None,
		read_model_path = None,
		read_replay_buffer_path = None,
		convert_slim = False,
		with_distillation = False,
		use_slim = False,
		action_noise = None,
		use_cuda = True,
		name = 'Model',
	)


	## MODIFIERS
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
				bounds_component = 'MapBounds',
				dz = dz,
				random = True,
				vertical = vertical,
			),
		],
		order = 'post',
		name = 'Spawner',
	)
	# EVALUATOR
	from others.evaluator import Evaluator
	# Evaluate model after each epoch and check curriculum learning
	Evaluator(
		goal_component = 'Goal',
		curriculum_percent = 0.8,
		distance_max = 100,
		level_up_distance = 5,
		name = 'Evaluator',
	)
	# SAVERS
	from modifiers.saver import Saver
	# save Train states and observations
	if instance_name in ['apolloW']:
		Saver(
			base_component = 'Environment',
			parent_method = 'end',
			track_vars = [
						'observations', 
						'states',
						],
			order = 'post',
			save_config = True,
			save_benchmarks = True,
			frequency = 1000,
			name='TrainEnvSaver',
		)
	# save model and replay buffer
	Saver(
		base_component = 'Model',
		parent_method = 'end',
		track_vars = [
					'model', 
					#'replay_buffer',
					],
		order = 'post',
		frequency = 1000,
		name='ModelSaver',
	)
	print('created new configuration')

# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
configuration.controller.stop()