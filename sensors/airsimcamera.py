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
    def sense(self):
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

    # creates a new observation object from passed in data
    def create_observation(self, data):
        image = Image(
            _data=data, 
            is_gray=self.is_gray,
        )
        return image


# junk:
'''
        
# if write_path is None will not save image, otherwise will save with '.png' appended to end of path
# ImageRequest is a class that gets passed as a parameter to simGetImages which returns an array of all image request returns
# transforms follow advice from https://microsoft.github.io/AirSim/image_apis/ and are adapted from airsim drone_env.py - transforms response from simGetImages() to an image
# RGB 3-band from ints
def transform_obs1(self, response):
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    return np.reshape(img1d, (response.height, response.width, 3))
# RGB 3-band to grayscale 1-band from ints
def transform_obs2(self, response):
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    img2d = np.reshape(img1d, (response.height, response.width, 3))
    image = Image.fromarray(img2d)
    im_gray = np.array(image.resize((response.width, response.height)).convert("L"))
    return im_gray.reshape((response.height, response.width, 1))
# grayscale 1-band from floats
def transform_obs3(self, response):
    img1d = np.array(response.image_data_float, dtype=np.float)
    img2d = np.reshape(img1d, (response.height, response.width))
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((response.width, response.height)).convert("L"))
    return im_final.reshape((response.height, response.width, 1))
# grayscale 1-band normalizes so far away depth are black and nearby are more white
def transform_obs4(self, response):
    img1d = np.array(response.image_data_float, dtype=np.float)
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (response.height, response.width))
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((response.width, response.height)).convert("L"))
    return im_final.reshape((response.height, response.width, 1))
# grayscale 1-band normalizes by min-max
def transform_obs5(self, response):
    img1d = np.array(response.image_data_float, dtype=np.float)
    img1d = 255 * (img1d - np.min(img1d)) / np.ptp(img1d)
    img2d = np.reshape(img1d, (response.height, response.width))
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((response.width, response.height)).convert("L"))
    return im_final.reshape((response.height, response.width, 1))
def sample(self, write_path=None):
    if write_path is None:
        write_path = 'sample/'

    responses = self.client.simGetImages([

        airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(2, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(3, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(4, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False),
        airsim.ImageRequest(0, airsim.ImageType.SurfaceNormals, False, False),

        airsim.ImageRequest(0, airsim.ImageType.Infrared, False, False),

        airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest(0, airsim.ImageType.DepthVis, True, False),
            
        airsim.ImageRequest(1, airsim.ImageType.OpticalFlow, True, False),
        airsim.ImageRequest(1, airsim.ImageType.OpticalFlowVis, True, False),
        airsim.ImageRequest(1, airsim.ImageType.DisparityNormalized, True, False),
    ])

    # Lidar script from https://github.com/microsoft/AirSim/blob/main/PythonClient/multirotor/drone_lidar.py
    for i in range(1,5):
        lidarData = self.client.getLidarData();
        if (len(lidarData.point_cloud) < 3):
            continue
        else:
            points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            pickle.dump(points, open(write_path + 'points' + str(i) + '.p', 'wb'))
            pickle.dump(lidarData, open(write_path + 'lidarData' + str(i) + '.p', 'wb'))
        time.sleep(1)

    imgs1 = [self.transform_obs1(response) for response in responses[0:7]]
    airsim.write_png(write_path + 'Scene_front_center.png', imgs1[0]) 
    airsim.write_png(write_path + 'Scene_front_right.png', imgs1[1]) 
    airsim.write_png(write_path + 'Scene_front_left.png', imgs1[2]) 
    airsim.write_png(write_path + 'Scene_bottom_center.png', imgs1[3]) 
    airsim.write_png(write_path + 'Scene_back_center.png', imgs1[4]) 
    airsim.write_png(write_path + 'Segmentation_front_center.png', imgs1[5]) 
    airsim.write_png(write_path + 'SurfaceNormals_front_center.png', imgs1[6]) 

    imgs2 = [self.transform_obs2(response) for response in responses[7:8]]
    airsim.write_png(write_path + 'Infrared_front_center.png', imgs2[0]) 
        
    imgs3 = [self.transform_obs3(response) for response in responses[8:14]]
    airsim.write_png(write_path + 'DepthPlanar_front_center.png', imgs3[0]) 
    airsim.write_png(write_path + 'DepthPerspective_front_center.png', imgs3[1]) 
    airsim.write_png(write_path + 'DepthVis_front_center.png', imgs3[2]) 
    airsim.write_png(write_path + 'OpticalFlow_front_center.png', imgs3[3]) 
    airsim.write_png(write_path + 'OpticalFlowVis_front_center.png', imgs3[4]) 
    airsim.write_png(write_path + 'DisparityNormalized_front_center.png', imgs3[5]) 
        
    imgs4 = [self.transform_obs4(response) for response in responses[8:14]]
    airsim.write_png(write_path + 'DepthPlanar_front_center_2.png', imgs4[0]) 
    airsim.write_png(write_path + 'DepthPerspective_front_center_2.png', imgs4[1]) 
    airsim.write_png(write_path + 'DepthVis_front_center_2.png', imgs4[2]) 
    airsim.write_png(write_path + 'OpticalFlow_front_center_2.png', imgs4[3]) 
    airsim.write_png(write_path + 'OpticalFlowVis_front_center_2.png', imgs4[4]) 
    airsim.write_png(write_path + 'DisparityNormalized_front_center_2.png', imgs4[5]) 
        
    imgs5 = [self.transform_obs5(response) for response in responses[8:14]]
    airsim.write_png(write_path + 'DepthPlanar_front_center_3.png', imgs5[0]) 
    airsim.write_png(write_path + 'DepthPerspective_front_center_3.png', imgs5[1]) 
    airsim.write_png(write_path + 'DepthVis_front_center_3.png', imgs5[2]) 
    airsim.write_png(write_path + 'OpticalFlow_front_center_3.png', imgs5[3]) 
    airsim.write_png(write_path + 'OpticalFlowVis_front_center_3.png', imgs5[4]) 
    airsim.write_png(write_path + 'DisparityNormalized_front_center_3.png', imgs5[5]) 

def sense(self, write_path=None):
    response = self.client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
    ])[0]
    img = self.transform_obs4(response)
    if write_path is not None:
        airsim.write_png(write_path + 'DepthPerspective_front_center_2.png', img) 
    return img

'''