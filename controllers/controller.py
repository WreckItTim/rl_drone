import utils
import os

# parent class for navigation mode
class Mode:    
    # parent constructor attaches drone instance to this mode instance and sets log vars
    def __init__(self, drone=None, log=False, log_path=None):
        self._drone = drone
        self._log = log
        self._log_path = log_path

    def run(self):
        raise NotImplementedError
    
    # issues command to drone
    # timestep is int representing the nth command issued (used for logging)
    # cmd is string with command name and params - see commands for parsing
    def command(self, timestep, cmd):
        cmd = cmd.lower()
        print('mode command:', cmd)
        executed = True
        timestep_path = None
        if self.log:
            utils.write_json(self.drone.commands, self.log_path + 'commands.json')
            timestep_path = self.log_path + 'timestep_' + str(timestep) + '/'
            os.mkdir(timestep_path)
        #try:
        if 'takeoff' in cmd:
            self.drone.takeOff()
        elif 'land' in cmd:
            self.drone.land()
        elif 'sample' in cmd:
            self.drone.sample(timestep_path)
        elif 'sense' in cmd:
            self.drone.sense(timestep_path)
        elif 'command' in cmd:
            drone_command = cmd.replace('command ', '')
            self.drone.command(drone_command)
        elif 'move' in cmd:
            parts = cmd.split(' ')
            self.drone.move(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        elif 'moveTo' in cmd:
            parts = cmd.split(' ')
            self.drone.moveTo(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        elif 'flip' in cmd:
            if ' ' in cmd:
                self.drone.flip(cmd.split(' ')[1])
            else:
                self.drone.flip()
        elif 'hover' in cmd:
            self.drone.hover()
        else:
            executed = False
            print('invalid command input')
        #except Exception as e:
        #    executed = False
        #    print('error during command', e)
        self._commands[timestep] = (cmd, executed)