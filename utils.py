import json
import os
from time import localtime, time
  
def path_to_list(path):
	cmd_list = ['takeoff']
	for point in path:
		cmd_list.append(f'moveTo {point[0]} {point[1]} {point[2]} {point[3]}')
	cmd_list.append('land')
	return cmd_list

def read_json(path):
	return json.load(open(path, 'r'))

def write_json(dic, path):
	json.dump(dic, open(path, 'w'), indent=2)

def move_to_string(point):
	return f'move {point[0]} {point[1]} {point[2]} {point[3]}'

def moveTo_to_string(point):
	return f'moveTo {point[0]} {point[1]} {point[2]} {point[3]}'

def error(msg):
	raise Exception(msg)

def get_time_stamp():
	secondsSinceEpoch = time()
	time_obj = localtime(secondsSinceEpoch)
	time_stamp = '%d_%d_%d_%d_%d_%d' % (
		time_obj.tm_year, time_obj.tm_mon, time_obj.tm_mday,  
		time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec
	)
	return time_stamp

def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory