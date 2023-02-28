import numpy as np
from threading import Thread
import socket
from drones.drone import Drone
from os import system
from component import _init_wrapper
import rl_utils as utils
import numpy as np
import math

# function that runs on another thread to constantly receive messages being sent from drone
def recv():
	while True: 
		try:
			response, ip = sock.recvfrom(1024)
			print('Tello Sent Message:', response)
		except Exception as e:
			continue

# TODO: silence opencv videocapture output
# TODO: improve feedback loop
# TODO: handle errors when issuing command - especially wind
# TODO: allow connecting to wifi for first time - with password
class Tello(Drone):

	@_init_wrapper
	def __init__(self, wifi_name = 'cloud', wifi_password = 'bustersword'):
		super().__init__()
		self._pos = np.array([0, 0, 0], dtype=float)
		self._speed = 100
		
	# if something goes wrong
	def stop(self):
		self.command('stop')

	# function to issue command to drone
	# for full list of commands see: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
	def command(self, msg):
		response = self._sock.sendto(msg.encode(encoding="utf-8"), self.address)
		print(f'command:{msg}', f'response:{response}')

	def reset(self, state=None):
		self._pos = np.array([0,0,0], dtype=float)
		self._yaw = 0
		self.take_off()

	# DUMMY
	def check_collision(self):
		return False

	def connect(self):
		super().connect()
		# connect to wifi
		'''
		print('Connecting to WiFi, send any key when complete and ready to continue...')
		system('cmd /c \"netsh wlan connect _name=' + self.wifi_name + '\"')
		x = input()
		'''
		ini = utils.prompt('send any key when connected to Tello via WiFi...')
		# open sockets to send/receive commands/stream to/from drone
		host = ''
		port = 9000
		locaddr = (host, port) 
		self.address = ('192.168.10.1', 8889)
		self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self._sock.bind(locaddr)
		self._recvThread = Thread(target=recv)
		self._recvThread.daemon = True
		self._recvThread.start()
		# establish link with drone
		response = self.command('command')
		# turn on streaming for camera output
		self.command('streamon')

	def disconnect(self):
		self._sock.close()
		
	# NEW: rotates along z-axis, yaw in radians [-pi, pi] offset from current yaw
	def rotate(self, yaw):
		yaw_deg = abs(math.degrees(yaw))
		if yaw > 0:
			# rotate clockwise
			self.command(f'cw {yaw_deg}')
		if yaw < 0:
			# rotate counterclockwise
			self.command(f'ccw {yaw_deg}')
		self._yaw += yaw


	# move to relative position, point [x,y,z] in m, speed in m/s
	def move(self, x, y, z, speed):
		# convert speed in m/s to cm/s
		#speed *= 100
		speed = 100
		# convert position from m to cm
		x, y, z = 100*x, 100*y, 100*z
		if speed < 10 or speed > 100:
			utils.error('speed out of range for tello')
		#if min(abs(x), abs(y), abs(z)) < 100 or max(abs(x), abs(y), abs(z)) > 500:
		#    utils.error('position out of range for tello')
		# have to move at max 500 cm increments
		def get_step(v):
			v2 = 0
			sign = np.sign(v)
			av = abs(v)
			if av >= 500:
				v2 = sign * 500
			else:
				v2 = sign * av 
			return v2
		
		while abs(x) > 0 or abs(y) > 0 or abs(z) > 0:
			x2 = get_step(x)
			y2 = get_step(y)
			z2 = get_step(z)
			x = np.sign(x) * max(0, np.abs(x) - 500)
			y = np.sign(y) * max(0, np.abs(y) - 500)
			z = np.sign(z) * max(0, np.abs(z) - 500)
			ini = utils.prompt(f'go {x2} {y2} {z2} {speed}?')
			if ini == 'y':
				self.command(f'go {x2} {y2} {z2} {speed}')
				self._pos += np.array([x2, y2, z2])
			elif ini == 'g':
				self.flip()
			else:
				self.command('emergency')
		
	# move to absolute position
	def move_to(self, point, speed, front_facing=False):
		position = get_position()
		x_diff, y_diff, z_diff = point[0]-position[0], point[1]-position[1], point[2]-position[2]
		if speed < 10 or speed > 100:
			utils.error('speed out of range for tello')
		if min(abs(x_diff), abs(y_diff), abs(z_diff)) < 100 or max(abs(x_diff), abs(y_diff), abs(z_diff)) > 1000:
			utils.error('position out of range for tello')
		self.command(f'go {x_diff} {y_diff} {z_diff} {speed}')
		self._pos = np.array([x, y, z])

	def flip(self, direction=None):
		if direction is None:
			direction = random.choice(['l', 'r', 'f', 'b'])
		response= self.command('flip ' +  direction)

	def take_off(self):
		response = self.command('takeoff')
		self._pos = np.array([0, 0, 0])

	def land(self):
		response = self.command('land')

	def get_position(self):
		return self._pos.tolist()

	def get_yaw(self):
		return self._yaw

	def hover(self):
		response = self.command('stop')

	def teleport(self, x, y, z, yaw, ignore_collision=True):
		pass