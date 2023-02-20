from run_base import create_base_components
import utils
import math

# create base components
continue_training = False
configuration = create_base_components(
	run_name = 'asnhVertCnn',
	continue_training = continue_training,
	controller_type = 'Train',
	airsim_release = 'AirSimNH',
	clock_speed = 16,
	include_z = True,
	flat = None,
)

# create other components
if not continue_training:
	# ACTOR
	actions=[
		'MoveForward',
		'MoveVertical',
		'Rotate',
		]
	from actors.continuousactor import ContinuousActor
	ContinuousActor(
		actions_components = actions,
		name='Actor',
	)

	# OBSERVER
	from observers.single import Single
	Single(
		sensors_components = ['GoalDistance', 'GoalOrientation', 'Moves'], 
		vector_length = 1 + 1 + len(actions),
		nTimesteps = 4,
		name = 'ObserverVector',
	)
	Single(
		sensors_components = ['DepthMap'], 
		is_image = True,
		image_height = 84, 
		image_width = 84,
		image_bands = 1,
		nTimesteps = 4,
		name = 'ObserverImage',
	)
	from observers.multi import Multi
	Multi(
		vector_observer_component = 'ObserverVector',
		image_observer_component = 'ObserverImage',
		name = 'Observer',
		)

	# CREATE MODEL
	from models.td3 import TD3
	TD3(
		environment_component = 'TrainEnvironment',
		policy = 'MultiInputPolicy',
		policy_kwargs = {'net_arch':[64,64]},
		buffer_size = 1000,
		learning_starts = 100,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		name='Model',
	)


utils.speak('configuration created!')

# CONNECT COMPONENTS
configuration.connect_all()

# view neural net archetecture
model_name = str(configuration.get_component('Model')._child())
sb3_model = configuration.get_component('Model')._sb3model
if 'dqn' in model_name:
	print(sb3_model.q_net)
	for name, param in sb3_model.q_net.named_parameters():
		print(name, param)
if 'ddpg' in model_name or 'td3' in model_name:
	print(sb3_model.critic)
	for name, param in sb3_model.critic.named_parameters():
		print(name, param)
utils.speak('all components connected. Send any key to continue...')
x = input()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
configuration.controller.stop()