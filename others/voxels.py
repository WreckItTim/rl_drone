from datastructs.datastruct import DataStruct
from component import _init_wrapper
import random
import numpy as np
import utils
import binvox as bv

# voxels is used to handle 3d/2d map representation of objects
# it is used by several software for 3d visualization
# you can create a .binvox file with voxels info from an AirSim/Unreal map with simCreateVoxelGrid()
# voxel is a compressed 3d boolean array with indexed positions to surfaces of all objects, True=surface, False=empty
# currently supported file formats are: .binvox
class Voxels(DataStruct):
	# example:
	# x_length = 200, y_length = 200, z_length = 100, resolution = 0.5, center = [0, 0, 0]
	# x-axis would have 200/0.5=400 grid points, centered around origin with meter range [-100, 100)
	# y-axis would have 200/0.5=400 grid points, centered around origin with meter range [-100, 100)
	# z-axis would have 100/0.5=200 grid points, centered around origin with meter range [-50, 50)
	# AirSim/Drone coordinates will place the y-axis along the voxel 0-axis, x-axis along 1-axis, -1*z along 2-axis
	# Voxels will normalize to a cube with length, height, width = 1; centered around center
	# Voxels file has translation to center grid, and scale to normalize it, 
	# indicies are cacluated like... xi = int((x/resolution[1] - translation[1]))
	# where resolution[1] = absolute(translation[1]) * 2 * scale
	# constructor only handles reading in from file at the moment
	@_init_wrapper
	def __init__(self,
			  # path to read/write voxels (compresed 3d representation of surface objects)
			  # will build absolute path from this
			  relative_path:str,
			  map_component,
			  make_new = True,
			  # can leave below blank (if reading from file rather than making new)
			  floor_z = None, # index of largest z-axis index for surface of floor from voxels 
				# None will auto find floor_dim from origin, assumes there is no object there in map upon connect() (excluding drone)
			  # voxel params if make new voxels (else these are set from read)
			  center = [0,0,0], # in meters
			  resolution = 1, # in meters
			  x_length = 200, # total x-axis meters (split around center)
			  y_length = 200, # total y-axis  meters (split around center)
			  z_length = 100, # total z-axis  meters (split around center)
			  ):
		super().__init__()
		self._map_2d = None
		
	
	def connect(self):
		super().connect()
		if self.make_new:
			self._map.make_voxels(
				absolute_path = utils.get_global_parameter('absolute_path') + self.relative_path,
				center = self.center,
				resolution = self.resolution,
				x_length = self.x_length,
				y_length = self.y_length,
				z_length = self.z_length,
				)
		self.read_voxels(
				utils.get_global_parameter('absolute_path') + self.relative_path
				)

	def get_x_length(self):
		return self._voxels.data.shape[1] 

	def get_y_length(self):
		return self._voxels.data.shape[0] 

	def get_z_length(self):
		return self._voxels.data.shape[2] 
			
	# read voxels (3d object map) from file
	# this is not needed (I used it to handle invalid spawns and objective points)
	def read_voxels(self, path):
		self._voxels = bv.Binvox.read(path, 'dense')
		scale = self._voxels.scale 
		trans = self._voxels.translate
		res = (np.absolute(trans)) * 2 * scale
		# drone cooridnates use +x as straight ahead, +y as to the right, and -z as up, relative to origin
		# voxels save in fundamental (x,y,z) euclidian space
		# keep this mind with conversion functions below
		# create functions to convert between drone position and voxel indicies
		self._x_to_xi = lambda x : int((x/res[1] - trans[1]))
		self._y_to_yi = lambda y : int((y/res[0] - trans[0]))
		self._z_to_zi = lambda z : int(((-1*z)/res[2] - trans[2]))
		self._xi_to_x = lambda xi : float((xi + trans[1])*res[1])
		self._yi_to_y = lambda yi : float((yi + trans[0])*res[0])
		self._zi_to_z = lambda zi : float(-1*(zi + trans[2])*res[2])
		print('loaded voxels from file')

	def get_map_2d(self):
		if self._map_2d is not None:
			return self._map_2d
		# turn 3d voxels map into 2d aerial view
		# map_3d is 3d array, elements are true where surface of objects are
		map_3d = self._voxels.data 
		x_len = map_3d.shape[1]
		y_len = map_3d.shape[0]
		z_len = map_3d.shape[2]
		# map_2d will be aerial view, with no z axis
		self._map_2d = np.full((x_len, y_len), False, dtype=bool)
		# we need to get the floor from the voxels
		# note that this only works for level floors
		# TODO: better conversion here considering uneven floors
		if self.floor_z is None:
			# get floor height (ASSUMES no object at 0,0) - messy I know, need a better solution
			origin = map_3d[int(map_3d.shape[0]/2), int(map_3d.shape[1]/2), :]
			_floor_z = max([i for i, is_obj in enumerate(origin) if is_obj])
		else:
			_floor_z = self.floor_z
		# set map 2d points
		for i in range(y_len):
			for j in range(x_len):
				for k in range(z_len):
					if map_3d[i, j, k] and k > _floor_z:
						# pad nearby cells
						for h in range(-1, 2, 1):
							for v in range(-1, 2, 1):
								if j+h > 0 and j+h < x_len:
									if i+v > 0 and i+v < y_len:
										self._map_2d[j+h, i+v] = True
						#self._map_2d[j, i] = True
						break
		return self._map_2d

	def write_2d_map(self, path=None):
		map_2d = self.get_map_2d()
		np.save(path, map_2d)