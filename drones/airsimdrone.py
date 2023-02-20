# drone launched through airsim api - requires an airsim relase map
import setup_path # need this in same directory as python code for airsim
import airsim
from drones.drone import Drone
from component import _init_wrapper
import math
import numpy as np

class AirSimDrone(Drone):
	@_init_wrapper
	def __init__(self,
			  airsim_component,
			  # the below boolean is to fix AirSim's y-drift problem
			  # see this GitHub ticket, with youtube video showing problem:
			  # https://github.com/microsoft/AirSim/issues/4780
			  drift_stop_gap = True,
			  ):
		super().__init__()
		self._speed = 2
		
	# check if has collided
	def check_collision(self):
		collision_info = self._airsim._client.simGetCollisionInfo()
		has_collided = collision_info.has_collided
		# collisions with floor are sometimes not registered
		if not has_collided:
			position = self.get_position()
			has_collided = position[2] > -1
		return has_collided 

	# resets on episode
	def reset(self, state=None):
		self._airsim._client.reset()
		self._airsim._client.enableApiControl(True)
		self._airsim._client.armDisarm(True)
		self.take_off()
		self.check_collision()

	def take_off(self):
		# take-off has some issues in airsim (sometimes the move after takeoff will fall to ground)
		# also prints outs lookahead values to console some times 
		# for w/e reason it is more stable to send command to fly up rather than using takeoff
		#self._airsim._client.takeoffAsync().join()
		self._airsim._client.moveByVelocityAsync(0, 0, -1, 2).join()
		'''
		self._airsim._client.setVelocityControllerGains(
			velocity_gains=airsim.VelocityControllerGains(
				airsim.PIDGains(0.2,0,0),
				airsim.PIDGains(0.2,0,0),
				airsim.PIDGains(2,2,0),
				)
			)
		default gains are:
		airsim.PIDGains(0.2,0,0),
		airsim.PIDGains(0.2,0,0),
		airsim.PIDGains(2,2,0),
		'''

	# returns state from client
	def get_state(self):
		return self._airsim._client.getMultirotorState()

	# if something goes wrong
	def stop(self):
		self.hover()
	
	def disconnect(self, state=None):
		pass

	# TODO: having problems with it landing sometimes - if done right after a move() command
	def land(self):
		self._airsim._client.landAsync().join()
	
	'''
	# OLD: move with speed for duration (unstable)
	def move(self, x_speed, y_speed, z_speed, duration):
		# this if statement is for a crazy no-good high-overhead stop gap
		# to temp-fix an issue with AirSim that adds drift when drone is facing y-axis
		if self.drift_stop_gap:
			yaw_deg = math.degrees(self.get_yaw())
			safe_yaw = 0 if abs(yaw_deg) < 90 else 180
			self._airsim._client.rotateToYawAsync(safe_yaw, timeout_sec=4, margin=1).join()

		# make actual movement
		self._airsim._client.moveByVelocityAsync(x_speed, y_speed, z_speed, duration).join()
		has_collided = self.check_collision()

		# undo stop_gap
		if self.drift_stop_gap and not has_collided:
			self._airsim._client.rotateToYawAsync(yaw_deg, timeout_sec=4, margin=1).join()
			
		return has_collided
	'''	
	# NEW: move to relative position with static speed
	def move(self, x_rel, y_rel, z_rel, duration=None):
		# this if statement is for a crazy no-good high-overhead stop gap
		# to temp-fix an issue with AirSim that adds drift when drone is facing y-axis
		if self.drift_stop_gap:
			yaw_deg = math.degrees(self.get_yaw())
			safe_yaw = 0 if abs(yaw_deg) < 90 else 180
			self._airsim._client.rotateToYawAsync(safe_yaw, timeout_sec=4, margin=1).join()

		# make actual movement
		current_position = self.get_position()
		target_position = np.array(current_position) + np.array([x_rel, y_rel, z_rel])
		self._airsim._client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], self._speed, timeout_sec=4).join()
		has_collided = self.check_collision()

		# undo stop_gap
		if self.drift_stop_gap and not has_collided:
			self._airsim._client.rotateToYawAsync(yaw_deg, timeout_sec=4, margin=1).join()
			
		return has_collided
	
	# move to absolute position
	def move_to(self, x, y, z, speed):
		self._airsim._client.moveToPositionAsync(x, y, z, speed).join()
	
	# teleports to position (ignores collisions), yaw in radians
	def teleport(self, x, y, z, yaw, ignore_collision=True):
		pose = airsim.Pose(
			airsim.Vector3r(x, y, z), 
			airsim.to_quaternion(0, 0, yaw)
		)
		self._airsim._client.simSetVehiclePose(pose, ignore_collision=ignore_collision)

	# OLD: rotates along z-axis, yaw_rate in deg/sec duration in sec  (unstable)
	#def rotate(self, yaw_rate, duration):
		#self._airsim._client.rotateByYawRateAsync(yaw_rate, duration).join()
	# NEW: rotates along z-axis, yaw_rate in deg offset from current yaw
	def rotate(self, yaw_deg, duration=None):
		current_yaw = math.degrees(self.get_yaw())
		target_yaw = current_yaw + yaw_deg
		self._airsim._client.rotateToYawAsync(target_yaw, timeout_sec=4, margin=1).join()
		has_collided = self.check_collision()
		return has_collided

	# get (x, y, z) positon, z is negative for up, x is positive for forward, y is positive for right (from origin)
	def get_position(self):
		pos = self._airsim._client.getMultirotorState().kinematics_estimated.position
		return [pos.x_val, pos.y_val, pos.z_val]

	# get rotation about the z-axis (yaw), returns in radians
	def get_yaw(self):
		q = self._airsim._client.getMultirotorState().kinematics_estimated.orientation
		pitch, roll, yaw = airsim.to_eularian_angles(q)
		return yaw

	def hover(self):
		self._airsim._client.hoverAsync().join()
