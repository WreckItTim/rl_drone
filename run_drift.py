from run_base import create_base_components
import utils
import math

# create base components
continue_training = False
flat_cols = [16, 32, 52, 68, 84]
flat_rows = [42]
configuration = create_base_components(
	run_type = 'drift', 
	run_extra = 'run2',
	drift_stop_gap = False,
	continue_training = continue_training,
	controller_type = 'AirSimChecks',
	include_z = False,
	flat_cols = flat_cols,
	flat_rows = flat_rows,
)

# create other components
if not continue_training:
	
	# ACTOR
	actions=[
		'MoveForward',
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
		sensors_components = ['GoalDistance', 'GoalOrientation', 'FlattenedDepth', 'Moves'], 
		vector_length = 1 + 1 + len(flat_cols)*len(flat_rows) + len(actions),
		nTimesteps = 4,
		name = 'Observer',
	)

	# CREATE MODEL
	from models.td3 import TD3
	TD3(
		environment_component = 'TrainEnvironment',
		policy = 'MlpPolicy',
		policy_kwargs = {'net_arch':[64,64]},
		buffer_size = 1000,
		learning_starts = 100,
		tensorboard_log = utils.get_global_parameter('working_directory') + 'tensorboard/',
		overide_memory = True, # memory benchmark on
		name='Model',
	)

	# ALTITUDE ADJUSTER (for horizontal motion, 
		# since moving forward adds drift up)
	from modifiers.altadjust import AltAdjust
	AltAdjust(
		base_component = 'Actor',
		parent_method = 'step',
		drone_component = 'Drone',
		order = 'post',
		name = 'Evaluator',
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
