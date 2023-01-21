# abstract class used to handle maps - where the drone is flying in
# this is the part that connects the program to the real application
from component import Component


class Map(Component):
	# constructor
	def __init__(self):
		pass

	# uses voxels to check if position is at 2d position of an object
	def at_object_2d(self, x, y):
		if self._voxels is None:
			# no other check so far if _voxels is empty
			return False
		xi = self._voxels._x_to_xi(x)
		yi = self._voxels._y_to_yi(y)
		# out of bounds
		if (xi < 0 or yi < 0
			or xi >= self._voxels.get_x_length()
			or yi >= self._voxels.get_y_length()
		):
			return False
		# check bools for object in that position
		map_2d = self._voxels.get_map_2d()
		return map_2d[xi, yi]

	def connect(self):
		super().connect()