# drone launched through airsim api - requires an airsim relase map
import setup_path # need this in same directory as python code for airsim
import airsim
from drones.drone import Drone
from math import sqrt
from component import _init_wrapper

class AirSimDrone(Drone):
    @_init_wrapper
    def __init__(self):
        super().__init__()
        self._client = None
        
    # check if has collided
    def check_collision(self):
        collision_info = self._client.simGetCollisionInfo()
        has_collided = collision_info.has_collided
        return has_collided 

    # resets on episode
    def reset(self):
        self._client.reset()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self.check_collision()

    # if something goes wrong
    def stop(self):
        self.hover()

    def connect(self):
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self.reset() # this seems repetitive but needed to reset state info
    
    def disconnect(self):
        if self._client is not None:
            self._client.armDisarm(False)
            self._client.reset()
            self._client.enableApiControl(False)
            self._client = None

    # will take off, move to a near by point, then land
    # WARNING TAKE SAFETY PRECAUTIONS
    def test(self):
        print('drone take off...')
        self.take_off()
        print('drone move(1, -1, 1, 4)...')
        self.move(2, 2, 2, 2)
        print('drone hover...')
        self.hover()

    def take_off(self):
        self._client.takeoffAsync().join()
        self.check_collision()

    # TODO: having problems with it landing sometimes - if done right after a move() command
    def land(self):
        self._client.landAsync().join()
    
    def move(self, x_distance, y_distance, z_distance, speed, front_facing=True):
        distance = sqrt(x_distance**2 + y_distance**2 + z_distance**2)
        duration = distance / speed
        x_speed = x_distance/duration
        y_speed = y_distance/duration
        z_speed = z_distance/duration
        drive_train = 1 if front_facing else 0
        self._client.moveByVelocityAsync(x_speed, y_speed, -1*z_speed, duration, drivetrain=drive_train).join()
    
    def move_to(self, x_position, y_position, z_position, speed, front_facing=True):
        drive_train = 1 if front_facing else 0
        self._client.moveToPositionAsync(x_position, y_position, -1*z_position, speed, drivetrain=drive_train).join()
    
    def get_position(self):
        return self._client.getMultirotorState().kinematics_estimated.position.to_numpy_array()

    def hover(self):
        self._client.hoverAsync().join()