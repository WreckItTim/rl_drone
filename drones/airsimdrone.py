# drone launched through airsim api - requires an airsim relase map
import setup_path # need this in same directory as python code for airsim
import airsim
import utils
from drones.drone import Drone
import math
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

	# if something goes wrong
	def stop(self):
		self.hover()

	def connect(self):
		super().connect()
		self._client = airsim.MultirotorClient(
			ip=utils.get_global_parameter('LocalHostIp'),
			port=utils.get_global_parameter('ApiServerPort'),
										 )
		self._client.confirmConnection()
		self._client.enableApiControl(True)
		self._client.armDisarm(True)
		self.reset() # this seems repetitive but needed to reset state info
	
	def disconnect(self):
		pass

	def take_off(self):
		# this is just smoother and more reliable than using take_off
		self._client.moveByVelocityAsync(0, 0, -1, 2)
		self.check_collision()
		#while self._client.getMultirotorState().landed_state == 0:
		#	self._client.takeoffAsync().join()

	# TODO: having problems with it landing sometimes - if done right after a move() command
	def land(self):
		self._client.landAsync().join()
	
	# move to relative position
	def move(self, x_speed, y_speed, z_speed, duration):
		self._client.moveByVelocityAsync(x_speed, y_speed, z_speed, duration).join()
	
	# move to absolute position
	def move_to(self, x, y, z, speed):
		self._client.moveToPositionAsync(x, y, z, speed).join()
	
	# teleports to position (ignores collisions), yaw in radians
	def teleport(self, x, y, z, yaw):
		pose = airsim.Pose(
			airsim.Vector3r(x, y, z), 
			airsim.to_quaternion(0, 0, yaw)
		)
		self._client.simSetVehiclePose(pose, ignore_collision=True)

	# rotates along z-axis, yaw_rate in deg/sec duration in sec
	def rotate(self, yaw_rate, duration):
		self._client.rotateByYawRateAsync(yaw_rate, duration).join()

	# get (x, y, z) positon, z is negative for up, x is positive for forward, y is positive for right (from origin)
	def get_position(self):
		pos = self._client.getMultirotorState().kinematics_estimated.position
		return [pos.x_val, pos.y_val, pos.z_val]

	# get rotation about the z-axis (yaw), returns in radians
	def get_yaw(self):
		q = self._client.getMultirotorState().kinematics_estimated.orientation
		pitch, roll, yaw = airsim.to_eularian_angles(q)
		return yaw

	def hover(self):
		self._client.hoverAsync().join()