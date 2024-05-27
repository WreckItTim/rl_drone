

# set wifi _name BEFORE playing with tello and run this script BEFORE using to be ready to kill with a key send
wifi_component = ''
import os
import socket
import time
        
host = ''
port = 9000
locaddr = (host, port) 
address = ('192.168.10.1', 8889)

while(True):
    print('initialized. send any key to hault drone')
    x = input()

    '''
    # connect to wifi
    print('attempting to connect to wifi...')
    while(True):
        try:
            time.sleep(10)
            os.system('cmd /c \"netsh wlan connect _name={' + self.wifi_component + '}\"')
            break
        except Exception as e:
            continue
    '''
 
    # open sockets to send commands
    print('attempting to connect sockets...')
    while(True):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(locaddr)
            break
        except Exception as e:
            continue

    # tell drone to listen for command
    while(True):
        try:
            print('Sent command, returned value:', sock.sendto('command'.encode(encoding="utf-8"), address))
            break
        except Exception as e:
            continue

    # send command to abort
    while(True):
        try:
            #print('Sent stop, returned value:', sock.sendto('stop'.encode(encoding="utf-8"), address))
            #print('Sent land, returned value:', sock.sendto('land'.encode(encoding="utf-8"), address))
            print('Sent emergency, returned value:', sock.sendto('emergency'.encode(encoding="utf-8"), address))
            break
        except Exception as e:
            continue

    print('drone killed! Going back to begining of cycle, to kill again when needed...')
    