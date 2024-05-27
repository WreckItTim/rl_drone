from controllers.controller import Controller
from component import _init_wrapper
import numpy as np
import math
import rl_utils as utils
import pickle

# collects data by collecting observations and calculating rewards 
	# given a set of static path x,y,z,yaw points
	# outputs a randomly initialized model
	# outputs 
	# this is brute forced to use an action space: move_forward, rotate, move_vertical
class Data(Controller):
	# constructor
	@_init_wrapper
	def __init__(self, 
				 model_component,
				 environment_component, # takes point at each step
				 drone_component,
				 ):
		super().__init__()

	# runs control on components
	def run(self):
		mdl_path = utils.get_global_parameter('working_directory') + 'random_init_model.zip'
		self._model.save_model(mdl_path)
		# numpy file is 3d:
			# axis 0 is path number
			# axis 1 is step number on path
			# axis 2 contains points
			# example:
			# path[0][0][:] = spawn point of first path
			# path[0][1][:] = goal point of first path
			# path[-1][-1][:] = end point of last path
		parent_dir = 'local/pretrain/'
		files = [
			'paths_horizontal_test_part0.p',
			'paths_horizontal_val_part0.p',
			'paths_horizontal_train_part0.p',
			#'paths_vertical_test_part0.p',
			#'paths_vertical_test_part1.p',
			#'paths_vertical_val_part0.p',
			#'paths_vertical_val_part1.p',
			#'paths_vertical_train_part0.p',
			#'paths_vertical_train_part1.p',
			#'paths_vertical_train_part2.p',
			#'paths_vertical_train_part3.p',
		]
		for fname in files:
			file_path = parent_dir + fname
			file = open(file_path, 'rb')
			paths = pickle.load(file)
			#points = np.load(self.points_file_path)
			nPaths = len(paths)
			for path_idx in range(nPaths):
				utils.speak(f'on path {path_idx} of {nPaths} from {fname}')
				# get path
				path = paths[path_idx]
				nPoints = len(path)
				# check end result
				#if path[-1][1] >= 10_000:
				#	continue
				# spawn at first point
				last_point = np.array(path[0].copy())
				# get goal
				goal_point = path[-2].copy()
				# reset env
				self._environment.reset(state={
					'spawn_to':list(last_point.copy()),
					'goal_at':list(goal_point.copy()),
					'fname':fname,
					})
				# move to each point in path
				for point_idx in range(1, nPoints-1):
					this_point = np.array(path[point_idx])
					distance_vector = this_point - last_point
					yaw_1_2 = math.atan2(distance_vector[1], distance_vector[0])
					# face next point
					self._drone.teleport(
						last_point[0], last_point[1], last_point[2],
						yaw_1_2, ignore_collision=True)
					yaw = this_point[3] - yaw_1_2
					if yaw > math.pi:
						yaw -= 2*math.pi
					elif yaw < -1 * math.pi:
						yaw += 2*math.pi
					forward = np.linalg.norm(distance_vector[:2])
					rl_output = np.array([forward/10, yaw/math.pi, distance_vector[2]/10], dtype=float)
					obs, reward, done, info = self._environment.step(rl_output)
					last_point = this_point.copy()
					if done:
						self._environment.end()
						break
				if not done:
					self._environment.end()