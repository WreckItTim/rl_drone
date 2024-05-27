# takes an image on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
from observations.image import Image
import numpy as np
from component import _init_wrapper
import rl_utils as utils
import cv2

# see https://microsoft.github.io/AirSim/image_apis/
class AirSimCamera(Sensor):
	
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
	
	# camera_view values:
		# 'front_center' or '0'
		# 'front_right' or '1'
		# 'front_left' or '2'
		# 'bottom_center' or '3'
		# 'back_center' or '4'
	# image_type values:
		# Scene = 0, 
		# DepthPlanar = 1, 
		# DepthPerspective = 2,
		# DepthVis = 3, 
		# DisparityNormalized = 4,
		# Segmentation = 5,
		# SurfaceNormals = 6,
		# Infrared = 7,
		# OpticalFlow = 8,
		# OpticalFlowVis = 9
	# constructor
	@_init_wrapper
	def __init__(self, 
			  airsim_component,
			  camera_view='0', 
			  image_type=2, 
			  as_float=True, 
			  compress=False, 
			  is_gray=True,
			  transformers_components=None,
			  offline = False,
			  ):
		super().__init__(offline)
		self._image_request = airsim.ImageRequest(camera_view, image_type, as_float, compress)
		if image_type in [1, 2, 3, 4]:
			self.is_gray = True
		# wrap all methods with crash handler
		# for method in dir(self):
		# 	if callable(getattr(self, method)) and method[0] != '_':
		# 		setattr(self, method, self._crash_handler(getattr(self, method)))

	def create_obj(self, data):
		observation = Image(
			_data=data, 
			is_gray=self.is_gray,
		)
		return observation

	# takes a picture with camera
	def step(self, state=None):
		img_array = []
		while len(img_array) <= 0: # loop for dead images (happens some times)
			response = self._airsim._client.simGetImages([self._image_request])[0]
			if self.as_float:
				np_flat = np.array(response.image_data_float, dtype=float)
			else:
				np_flat = np.fromstring(response.image_data_uint8, dtype=np.uint8)
			if self.is_gray:
				img_array = np.reshape(np_flat, (response.height, response.width))
				if len(img_array) > 0:
					# make channel-first
					img_array = np.expand_dims(img_array, axis=0)
			else:
				img_array = np.reshape(np_flat, (response.height, response.width, 3))
				if len(img_array) > 0:
					# make channel-first
					img_array = np.moveaxis(img_array, 2, 0)
		observation = self.create_obj(img_array)
		transformed = self.transform(observation)
		return transformed