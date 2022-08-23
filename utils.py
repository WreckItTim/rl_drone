import json
import os
from time import localtime, time
  

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


global_parameters = {}
def read_global_parameters(path = 'global_parameters.json'):
	global_parameters.update(read_json(path))
	from models.model import Model
	Model.model_overwrite_warning = global_parameters['model_overwrite_warning']

def write_global_parameters(path = 'global_parameters.json'):
	write_json(global_parameters, path)

def set_global_parameter(key, value):
	if key not in global_parameters:
		warning('key', key, 'not in global parameters')
	else:
		global_parameters[key] = value

def get_global_parameter(key):
	if key not in global_parameters:
		error('key', key, 'not in global parameters')
	else:
		return global_parameters[key]


def prompt(msg):
	print(msg)
	return str(input()).lower()

def error(msg):
	raise Exception('ERROR:', msg)

def warning(msg):
	print('WARNING:', msg)

def alert(msg, key):
	user_input = prompt('ALERT:', msg)
	if user_input == 'stop':
		set_global_parameter(key, False)