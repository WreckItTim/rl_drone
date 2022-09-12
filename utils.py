import json
from time import localtime, time
import math

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
	qz = nmath.sin(yaw/2)
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