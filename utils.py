import json
from time import localtime, time
import math
import os

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

def setup(working_directory):
	set_operating_system()
	set_read_write_paths(working_directory = working_directory)

def set_operating_system():
	import platform
	OS = platform.system()
	set_global_parameter('OS', OS)
	speak('detected operating system:', OS)
	
# end all folder paths with /
def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory

# set up folder paths for file io
def set_read_write_paths(working_directory):
	# make temp folder if not exists
	if not os.path.exists('temp/'):
		os.makedirs('temp/')
	# make working directory if not exists
	working_directory = fix_directory(working_directory)
	if not os.path.exists(working_directory):
		os.makedirs(working_directory)
	# save working directory path to global_parameters to be visible by all 
	set_global_parameter('working_directory', working_directory) # relative to repo
	# absoulte path on local computer to repo
	set_global_parameter('absolute_path',  os.getcwd() + '/')

# set up controller to run configuration on
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
	speak(controller_type, 'CONTROLLER')
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
			total_timesteps = total_timesteps,
			callback = None,
			log_interval = -1,
			tb_log_name = tb_log_name,
			continue_training = continue_training,
			)
	# evaluate will read in a trained model and evaluate on given environment
	elif controller_type == 'evaluate':
		from controllers.evaluaterl import EvaluateRL
		controller = EvaluateRL(
			evaluator_component = evaluator_component,
			)
	else:
		from controllers.empty import Empty
		controller = Empty()
	return controller


# GLOBAL PARAMS
global_parameters = {}
def read_global_parameters(path = 'global_parameters.json'):
	global_parameters.update(read_json(path))

def write_global_parameters(path = 'global_parameters.json'):
	write_json(global_parameters, path)

def del_global_parameter(key):
	if key in global_parameters:
		del global_parameters[key]

def set_global_parameter(key, value):
	global_parameters[key] = value

def get_global_parameter(key):
	if key not in global_parameters:
		return None
	else:
		return global_parameters[key]


# COMMUNICATE WITH USER
global_log = []
def add_to_log(msg):
	global_log.append(msg)
	print_global_log()
def print_global_log():
	file = open('log.txt', 'w')
	for item in global_log:
		file.write(item + "\n")
	file.close()

def speak(msg):
	add_to_log(msg)
	print(msg)

def prompt(msg):
	speak(msg)
	return get_user_input()

def get_user_input():
	return input()

def error(msg):
	add_to_log(msg)
	raise Exception('ERROR:', msg)

def warning(msg):
	speak('WARNING:', msg)



# STOPWATCH
# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class StopWatch:
  def __init__(self):
    self.start_time = time.time()
    self.last_time = self.start_time
    self.laps = []
  def lap(self):
    this_time = time.time()
    delta_time = this_time - self.last_time
    self.laps.append(delta_time)
    self.last_time = this_time
    return delta_time
  def stop(self):
    self.stop_time = time.time()
    self.delta_time = self.stop_time - self.start_time
    return self.delta_time