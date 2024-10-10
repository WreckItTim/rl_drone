from others.other import Other
from component import _init_wrapper
import rl_utils as utils

# rooftops handels a preprocessed data structure that is a dictionary:
	# rooftops[x][y] = z    where z is the highest collidable surface at that x,y coordinate
	# x, y, z are all descritized to 1m resolution cubes
# this is usefull when spawning random points to insure one does not spawn inside of an object
# if a preprocesssed rooftops object is not available for a map, that one can use Voxels if using AirSim
class Rooftops(Other):
	@_init_wrapper
	def __init__(self,
			  read_path, # pickle dictionary following structure outlined above
			  buffer = -2, # z-displacement buffer to add to roof to determine if collided with surface
			  ):
		super().__init__()
		
	
	def connect(self):
		super().connect()
		self._rooftops = utils.pk_read(self.read_path)

	def in_object(self, x, y, z):
		roof = self.get_roof(x, y)
		z = int(z)
		# check if spawned in object
		in_object = roof + self.buffer < z # drone z coords are negative uggghhh
		return in_object

	# will get lowest z-point without being inside object at given x,y
	# this includes the floor (which will be lowest point found)
	def get_roof(self, x, y):
		x, y = int(x), int(y)
		return self._rooftops[x][y]