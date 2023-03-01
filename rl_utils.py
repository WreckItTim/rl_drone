import json
from time import localtime, time
import math
import os
import shutil

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

def setup(write_parent, run_prefix):
	read_global_parameters()
	run_name = run_prefix + '_' + get_global_parameter('instance_name')
	set_global_parameter('run_name',  run_name)
	working_directory = write_parent + run_name + '/'
	set_read_write_paths(working_directory = working_directory)
	read_global_log()
	set_operating_system()

def set_operating_system():
	import platform
	OS = platform.system().lower()
	set_global_parameter('OS', OS)
	speak(f'detected operating system:{OS}')
	
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
	shutil.copyfile('train_eval.ipynb', working_directory + 'train_eval.ipynb')
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
	speak(f'CONTROLLER = {controller_type}')
	# debug mode will prompt user input for which component(s) to debug
	if controller_type == 'Debug':
		from controllers.debug import Debug
		controller = Debug(
			drone_component='Drone',
			)
	# train will create a new or read in a previously trained model
	# set continue_training=True to pick up where learning loop last saved
	# or set continue_training=False to keep weights but start new learning loop
	elif controller_type == 'Train':
		from controllers.train import Train
		controller = Train(
			model_component = model_component,
			environment_component = environment_component,
			total_timesteps = total_timesteps,
			callback = None,
			log_interval = -1,
			tb_log_name = tb_log_name,
			continue_training = continue_training,
			)
	# evaluate will read in a trained model and evaluate on given environment
	elif controller_type == 'Evaluate':
		from controllers.evaluate import Evaluate
		controller = Evaluate(
			evaluator_component = evaluator_component,
			)
	# checks will run drift checks
	elif controller_type == 'AirSimChecks':
		from controllers.airsimchecks import AirSimChecks
		controller = AirSimChecks(
			drone_component = 'Drone',
			actor_component = 'Actor',
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
	global_log.append(get_timestamp() + ': ' + msg)
	print_global_log()
def print_global_log():
	file = open(get_global_parameter('working_directory') + 'log.txt', 'w')
	for item in global_log:
		file.write(item + "\n")
	file.close()
def read_global_log():
	path = get_global_parameter('working_directory') + 'log.txt'
	if os.path.exists(path):
		with open(path) as file:
			for line in file:
				global_log.append(line.rstrip())

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
    self.start_time = time()
    self.last_time = self.start_time
    self.laps = []
  def lap(self):
    this_time = time()
    delta_time = this_time - self.last_time
    self.laps.append(delta_time)
    self.last_time = this_time
    return delta_time
  def stop(self):
    self.stop_time = time()
    self.delta_time = self.stop_time - self.start_time
    return self.delta_time
