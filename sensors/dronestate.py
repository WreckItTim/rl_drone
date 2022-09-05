from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper

# gets state information from drone
class DroneState(Sensor):

	# specify which state values to get
	@_init_wrapper
	def __init__(self,
				 drone_component,
				 get_yaw = True,
				 ):
		super().__init__()

	# get state information from drone
	def sense(self, logging_info=None):
		data = []
		if self.get_yaw:
			# fetch yaw
			yaw = self._drone.get_yaw()
			# normalize yaw between 0 and 1
			yaw_normalized = yaw / math.pi / 2
			data.append(yaw_normalized)
		observation = Vector(
			_data = data,
			names = logging_info,
		)
		return observation