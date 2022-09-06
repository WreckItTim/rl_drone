# takes an image on each observation
from sensors.sensor import Sensor
from observations.image import Image
import numpy as np
from cv2 import VideoCapture
from component import _init_wrapper

# see https://microsoft.github.io/AirSim/image_apis/
class PortCamera(Sensor):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  port='udp://0.0.0.0:11111', 
			  is_gray=False,
			  transformers_components=None,
			  ):
		super().__init__()
		self._camera = None

	def connect(self):
		super().connect()
		self._camera = VideoCapture(self.port)

	def disconnect(self):
		if self._camera is not None:
			self._camera.release()

	# takes a picture with camera
	def sense(self):
		ret = False
		while not ret:
			ret, img_array = self._camera.read()
		observation = Image(
			_data=img_array, 
			is_gray=self.is_gray,
		)
		return self.transform(observation)
