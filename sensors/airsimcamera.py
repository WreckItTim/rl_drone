# takes an image on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
from observations.image import Image
import numpy as np
from component import _init_wrapper

# see https://microsoft.github.io/AirSim/image_apis/
class AirSimCamera(Sensor):
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
	def __init__(self, camera_view='0', image_type=2, as_float=True, compress=False, is_gray=False):
		super().__init__()
		self._image_request = airsim.ImageRequest(camera_view, image_type, as_float, compress)
		self._client = None
		if image_type in [1, 2, 3, 4]:
			self.is_gray = True

	# resets on episode
	def reset(self):
		self._client.enableApiControl(True)
		self._client.armDisarm(True)

	def connect(self):
		super().connect()
		self._client = airsim.MultirotorClient()
		self._client.confirmConnection()

	# takes a picture with camera
	def sense(self, logging_info=None):
		response = self._client.simGetImages([self._image_request])[0]
		if self.as_float:
			np_flat = np.array(response.image_data_float, dtype=np.float)
		else:
			np_flat = np.fromstring(response.image_data_uint8, dtype=np.uint8)
		if self.is_gray:
			img_array = np.reshape(np_flat, (response.height, response.width))
		else:
			img_array = np.reshape(np_flat, (response.height, response.width, 3))
		image = Image(
			_data=img_array, 
			is_gray=self.is_gray,
		)
		return image