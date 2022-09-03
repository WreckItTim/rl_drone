# drone launched through airsim api - requires an airsim relase map
import setup_path # need this in same directory as python code for airsim
import airsim
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
		self.check_collision()

	# if something goes wrong
	def stop(self):
		self.hover()

	def connect(self):
		super().connect()
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

	def take_off(self):
		while self._client.getMultirotorState().landed_state == 0:
			self._client.takeoffAsync().join()
			#print('takeoff')

	# TODO: having problems with it landing sometimes - if done right after a move() command
	def land(self):
		self._client.landAsync().join()
	
	## move to relative position
	#def move(self, point, speed):
	#	x_distance, y_distance, z_distance = point[0], point[1], point[2]
	#	distance = math.sqrt(x_distance**2 + y_distance**2 + z_distance**2)
	#	duration = distance / speed
	#	x_speed = x_distance / duration
	#	y_speed = y_distance / duration
	#	z_speed = z_distance / duration
	#	duration = distance / speed
	#	#, yaw_mode={'is_rate':False,'yaw_or_rate':self._yaw_degrees}
	#	print(self._client.getMultirotorState())
	#	self._client.moveByVelocityAsync(x_speed, y_speed, z_speed, duration).join()
	#	print('move_ve', x_speed, y_speed, z_speed, duration, self.get_position())
	
	# move to relative position
	def move(self, point, speed):
		x_distance, y_distance, z_distance = point[0], point[1], point[2]
		x_position, y_position, z_position = self.get_position()
		self._prestate = self._client.getMultirotorState()
		self._client.moveToPositionAsync(x_position+x_distance, y_position+y_distance, z_position+z_distance, speed).join()
	
	# move to absolute position
	def move_to(self, point, speed):
		x_position, y_position, z_position = point[0], point[1], point[2]
		self._client.moveToPositionAsync(x_position, y_position, z_position, speed).join()
	
	# teleports to position (ignores collisions)
	def teleport(self, point):
		pose = self._client.simGetVehiclePose()
		pose.position.x_val = point[0] 
		pose.position.y_val = point[1]  
		pose.position.z_val = point[2]  
		self._client.simSetVehiclePose(pose, ignore_collision=True)

	# sets yaw (different than rotating)
	def set_yaw(self, yaw_degrees):
		self._client.rotateToYawAsync(yaw_degrees, timeout_sec=10).join()

	def get_position(self):
		pos = self._client.getMultirotorState().kinematics_estimated.position
		return [pos.x_val, pos.y_val, pos.z_val]

	# get rotation about the z-axis (yaw)
	def get_yaw(self, radians=True):
		q = self._client.getMultirotorState().kinematics_estimated.orientation
		w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val, 
		#yaw_radians = 2*math.acos(q.w_val)
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_radians = math.atan2(t3, t4)
		# TODO: these two quads or off for somereason (current solution is a brute force quick fix, find real problem later)
		#if yaw_radians < 0:
		#    yaw_radians += 2*math.pi
		if not radians:
			return math.degrees(yaw_radians)
		return yaw_radians

	def hover(self):
		self._client.hoverAsync().join()