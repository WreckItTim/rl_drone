import numpy as np
import cv2
from threading import Thread
import socket
from drones.drone import Drone
from os import system
from time import sleep
from component import _init_wrapper


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
  def __init__(self, wifi_name = 'cloud', wifi_password = 'bustersword', name=None):
    super().__init__()
    self.wifi_name = wifi_name
    self.wifi_password = wifi_password
    self._pos = np.array([0, 0, 0]).astype(int)

    # if something goes wrong
    def stop(self):
        self.command('stop')

  # will take off, move to a near by point, then land
  # WARNING TAKE SAFETY PRECAUTIONS
  def test(self):
    print('drone take off...')
    self.take_off()
    sleep(4)
    print('drone move(1, -1, 1, 4)...')
    while(True):
      self.move(1000, 1000, 1000, 100)
      x = input()
      if x == 'q':
        break
    sleep(4)
    print('drone hover...')
    self.hover()
    sleep(4)

  # function to issue command to drone
  # for full list of commands see: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
  def command(self, msg):
    response = self._sock.sendto(msg.encode(encoding="utf-8"), self.address)
    print(f'command:{msg}', f'response:{response}')

  def connect(self):
    # connect to wifi
    print('Connecting to WiFi, send any key when complete and ready to continue...')
    system('cmd /c \"netsh wlan connect name=' + self.wifi_name + '\"')
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

  def move(self, x, y, z, speed):
    self.command(f'go {x} {y} {z} {speed}')
    self._pos += np.array([x, y, z])

  def move_to(self, x, y, z):
    self.command(f'go {x-self._pos[0]} {y-self._pos[1]} {z-self._pos[2]} {self.speed}')
    self._pos = np.array([x, y, z])

  def flip(self, direction=None):
    if direction is None:
      direction = random.choice(['l', 'r', 'f', 'b'])
    sresponse= self.command('flip ' +  direction)

  def take_off(self):
    response = self.command('takeoff')
    self._pos = np.array([0, 0, 0])

  def land(self):
    response = self.command('land')

  def get_position(self):
    return self._pos

  def hover(self):
    response = self.command('stop')