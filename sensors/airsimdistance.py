# gets point distance from drone (set through airsim settings) on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
import rl_utils as utils
from observations.vector import Vector
import numpy as np
from component import _init_wrapper

class AirSimDistance(Sensor):
	
	# AirSim is unstable - need to handle crashes
	def _crash_handler(self, method):
		def _wrapper(*args, **kwargs):
			try_again = True
			while(try_again):
				try:
					method_output = method(*args, **kwargs)
					pos = self._airsim._client.getMultirotorState().kinematics_estimated.position
					q = self._airsim._client.getMultirotorState().kinematics_estimated.orientation
					pitch, roll, yaw = airsim.to_eularian_angles(q)
					self._last_pos_yaw = [pos.x_val, pos.y_val, pos.z_val, yaw]
					try_again = False
				except Exception as e: #msgpackrpc.error.TimeoutError as e:
					print(str(e) + ' caught from AirSim method ' + method.__name__)
					utils.add_to_log('CRASH ENCOUNTERED - ' + str(e) + ' caught from AirSimDrone method ' + method.__name__)
					self.handle_crash()
					utils.add_to_log('CRASH HANDLED')
			return method_output
		return _wrapper
	
	def handle_crash(self):
		self._airsim.connect(from_crash=True)
		self._airsim._client.reset()
		self._airsim._client.enableApiControl(True)
		self._airsim._client.armDisarm(True)
		self._airsim._client.moveByVelocityAsync(0, 0, -1, 2).join()
		x, y, z, yaw = self._last_pos_yaw
		pose = airsim.Pose(
			airsim.Vector3r(x, y, z), 
			airsim.to_quaternion(0, 0, yaw)
		)
		self._airsim._client.simSetVehiclePose(pose, ignore_collision=True)
		self._airsim._client.rotateByYawRateAsync(0, 0.001).join()
		self._airsim._client.moveByVelocityAsync(0, 0, 0, 0.001).join()
		collision_info = self._airsim._client.simGetCollisionInfo()

	# constructor
	@_init_wrapper
	def __init__(self,
			  airsim_component,
			  transformers_components=None,
			  offline = False,
			  ):
		super().__init__(offline)
		# wrap all methods with crash handler
		# for method in dir(self):
		# 	if callable(getattr(self, method)) and method[0] != '_':
		# 		setattr(self, method, self._crash_handler(getattr(self, method)))

	def create_obj(self, data):
		observation = Vector(
			_data=data, 
		)
		return observation

	# takes a picture with camera
	def step(self, state=None):
		distance = np.array(self._airsim._client.getDistanceSensorData().distance)
		observation = self.create_obj([distance])
		transformed = self.transform(observation)
		return transformed