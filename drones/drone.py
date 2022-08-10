from component import Component

class Drone(Component):
    # constructor
    def __init__(self):
        pass

    # updates number of collisions and returns same 
    def check_collision(self):
        raise NotImplementedError

    # setup anything that needs to be done to communicate with drone
    def connect(self):
        raise NotImplementedError

    # clean up any resources as needed when done with communication
    def disconnect(self):
        raise NotImplementedError
    
    # take off!
    def take_off(self):
        raise NotImplementedError

    # land!
    def land(self):
        raise NotImplementedError

    # moves to relative position at given speed (units defined within drone) 
    def move(self, x, y, z, speed, front_facing=True):
        raise NotImplementedError

    # moves to absolute position at given speed (units defined within drone) 
    def move_to(self, x, y, z, speed):
        raise NotImplementedError

    # return response from all active sensors
    def sense(self):
        raise NotImplementedError

    # issue command to drone via string
    def command(self):
        raise NotImplementedError

    # get current position of drone
    def get_position(self):
        raise NotImplementedError

    # enter hover mode
    def hover(self):
        raise NotImplementedError

    # reset to prepare for next run (episode for RL)
    def reset(self):
        raise NotImplementedError
        