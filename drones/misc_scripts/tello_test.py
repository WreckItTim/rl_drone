import cv2 # pip install opencv-python
import time
import socket
import threading

# open sockets to send/receive commands/stream to/from drone
host = ''
port = 9000
locaddr = (host,port) 
tello_address = ('192.168.10.1', 8889)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(locaddr)

# function to issue command to drone
# for full list of commands see: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
def command(msg, tello_address):
    msg = msg.encode(encoding="utf-8")
    print(msg, 'response:', sock.sendto(msg, tello_address))

# function that runs on another thread to constantly receive messages being sent from drone
def recv():
    while True:
        try:
            response, ip = sock.recvfrom(1024)
            print('Tello Response:', response)
        except Exception as e:
            pass

# create thread for receving messages
recvThread = threading.Thread(target=recv)
recvThread.daemon = True
recvThread.start()

# establish link with drone
command('command', tello_address)
time.sleep(1)


while (True):
    command(input(), tello_address)

'''
# turn on video stream
command('streamon', tello_address)
time.sleep(1)

# use opencv to read live video from drone
camera = cv2.VideoCapture('udp://0.0.0.0:11111')
time.sleep(1)

# loop to read and display video
# WARNING: make sure to press q to quit, so properly shuts down
sample_time = 0.2
last_grab = time.time()
time_start = last_grab
while(True):
    ret, frame = camera.read()
    if ret:
        if time.time() - last_grab >= sample_time:
            last_grab = time.time()
            cv2.imshow('Tello', frame)

    if time.time() - time_start > 4:
        command('takeoff', tello_address)
        time_start = 999999
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# cleanup
sock.close()
camera.release()
cv2.destroyAllWindows()
'''