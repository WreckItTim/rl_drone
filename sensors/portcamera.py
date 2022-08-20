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
    def __init__(self, port='udp://0.0.0.0:11111', is_gray=False):
        super().__init__()
        self._camera = None

    def connect(self):
        self._camera = VideoCapture(self.port)

    def disconnect(self):
        if self._camera is not None:
            self._camera.release()

    # takes a picture with camera
    def sense(self):
        ret = False
        while not ret:
            ret, img_array = self._camera.read()
        image = Image(
            _data=img_array, 
            is_gray=self.is_gray,
        )
        return image

    def test(self):
        print('taking picture...')
        image = self.sense()
        print('writing picture...')
        image.write('temp/')
        print('displaying picture...')
        image.display()