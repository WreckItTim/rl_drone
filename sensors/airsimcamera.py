# takes an image on each observation
from sensors.sensor import Sensor
import setup_path # need this in same directory as python code for airsim
import airsim
from observations.image import Image
from observations.array import Array
import numpy as np
from component import _init_wrapper
import rl_utils as utils

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
	def __init__(self, 
			  airsim_component,
			  camera_view='0', 
			  image_type=2, 
			  as_float=True, 
			  is_gray=False,
			  compress=False,
			  is_image = True,
			  transformers_components=None,
			  offline = False,
			  ):
		super().__init__(offline)
		self._image_request = airsim.ImageRequest(camera_view, image_type, as_float, compress)
		if image_type in [1, 2, 3, 4]:
			self.as_float = True
			self.is_gray = True
			self.is_image = False

	def create_obj(self, data):
		if self.is_image:
			observation = Image(
				_data=data, 
				is_gray=self.is_gray,
			)
		else:
			observation = Array(
				_data=data,
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
