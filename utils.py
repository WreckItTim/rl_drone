import json
from time import localtime, time
import math
import os
			
def _round(x, digits=2):
	if type(x) == list:
		for i, _x in enumerate(x):
			x[i] = round(_x, digits)
	else:
		x = round(x, digits)
	return x

def read_json(path):
	return json.load(open(path, 'r'))

def write_json(dictionary, path):
	json.dump(dictionary, open(path, 'w'), indent=2)

def get_timestamp():
	secondsSinceEpoch = time()
	time_obj = localtime(secondsSinceEpoch)
	timestamp = '%d_%d_%d_%d_%d_%d' % (
		time_obj.tm_year, time_obj.tm_mon, time_obj.tm_mday,  
		time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec
	)
	return timestamp

def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory

def debug_json(dictionary):
	print('debug json')
	for key in dictionary:
		if type(dictionary[key]) == dict:
			debug_json(dictionary[key])
		else:
			print(key, dictionary[key], type(dictionary[key]))

# gets yaw from position vector (relative to 0 origin)
def position_to_yaw(xyz_point):
	return math.atan2(xyz_point[1], xyz_point[0])
			
# assumes 0 pitch and roll, inputs yaw in radians
def yaw_to_quaternion(yaw):
	qx = 0
	qy = 0
	qz = math.sin(yaw/2)
	qw = math.cos(yaw/2)
	return qx, qy, qz, qw

# assumes 0 pitch and roll, so only inputs qz, returns yaw in radians
def quaternion_to_yaw(qz):
	yaw = 2 * math.arcsin(qz)
	return yaw

global_parameters = {}
def read_global_parameters(path = 'global_parameters.json'):
	global_parameters.update(read_json(path))
	from models.model import Model
	Model.model_overwrite_warning = global_parameters['model_overwrite_warning']

def write_global_parameters(path = 'global_parameters.json'):
	write_json(global_parameters, path)

def set_global_parameter(key, value):
	global_parameters[key] = value

def get_global_parameter(key):
	if key not in global_parameters:
		error(f'key {key} not in global parameters')
	else:
		return global_parameters[key]


def speak(msg):
	print(msg)

def prompt(msg):
	speak(msg)
	return str(get_user_input()).lower()

def get_user_input():
	return input()

def error(msg):
	raise Exception('ERROR:', msg)

def warning(msg):
	speak('WARNING:', msg)

def alert(msg, key):
	user_input = prompt('ALERT:', msg)
	if user_input == 'stop':
		set_global_parameter(key, False)
		write_global_parameters()

def set_operating_system():
	import platform
	OS = platform.system()
	set_global_parameter('OS', OS)
	print('detected operating system:', OS)

def set_read_write_paths(runs_path, run_name):
	# create working directory to read/write files to
	working_directory = runs_path + run_name + '/'
	# make temp folder if not exists
	if not os.path.exists('temp/'):
		os.makedirs('temp/')
	# make working directory if not exists
	if not os.path.exists(working_directory):
		os.makedirs(working_directory)
	# save working directory path to global_parameters to be visible by all 
	set_global_parameter('working_directory', working_directory) # relative to repo
	# absoulte path on local computer to repo
	set_global_parameter('absolute_path',  os.getcwd() + '/') # end all folder paths with /

def get_controller(controller_type, 
				   total_timesteps = 1_000_000,
				   continue_training = True,
				   model_component = 'Model',
				   environment_component = 'TrainEnvironment',
				   evaluator_component = 'Evaluator',
				   tb_log_name = 'run',
				   ):
	# create CONTROLLER - controls all components (mode)
	controller = None
	print(controller_type, 'CONTROLLER')
	# debug mode will prompt user input for which component(s) to debug
	if controller_type == 'debug':
		from controllers.debug import Debug
		controller = Debug(
			drone_component='Drone',
			)
	# train will create a new or read in a previously trained model
	# set continue_training=True to pick up where learning loop last saved
	# or set continue_training=False to keep weights but start new learning loop
	elif controller_type == 'train':
		from controllers.trainrl import TrainRL
		controller = TrainRL(
			model_component = model_component,
			environment_component = environment_component,
			evaluator_component = evaluator_component,
			total_timesteps = total_timesteps,
			callback = None,
			log_interval = -1,
			tb_log_name = tb_log_name,
			eval_env = None,
			eval_freq = -1,
			n_eval_episodes = -1,
			eval_log_path = None,
			continue_training = continue_training,
			)
	# evaluate will read in a trained model and evaluate on given environment
	elif controller_type == 'evaluate':
		from controllers.evaluaterl import EvaluateRL
		controller = EvaluateRL(
			evaluator_component = evaluator_component,
			)
	else:
		from controllers.controller import Controller
		controller = Controller()
	return controller