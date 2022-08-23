import numpy as np
import cv2
from threading import Thread
import socket
from drones.drone import Drone
from os import system
from time import sleep
from component import _init_wrapper
import utils

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
        self._pos = np.array([0, 0, 0]).astype(int)
        
    # if something goes wrong
    def stop(self):
        self.command('stop')

    # function to issue command to drone
    # for full list of commands see: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
    def command(self, msg):
        response = self._sock.sendto(msg.encode(encoding="utf-8"), self.address)
        print(f'command:{msg}', f'response:{response}')

    def connect(self):
        # connect to wifi
        print('Connecting to WiFi, send any key when complete and ready to continue...')
        system('cmd /c \"netsh wlan connect _name=' + self.wifi_component + '\"')
        x = input()
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

    # move to relative position
    def move(self, point, speed, front_facing=False):
        x, y, z = point[0], point[1], point[2]
        if speed < 10 or speed > 100:
            utils.error('speed out of range for tello')
        if min(abs(x), abs(y), abs(z)) < 100 or max(abs(x), abs(y), abs(z)) > 1000:
            utils.error('position out of range for tello')
        self.command(f'go {x} {y} {z} {speed}')
        self._pos += np.array([x, y, z])
        
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

    def hover(self):
        response = self.command('stop')