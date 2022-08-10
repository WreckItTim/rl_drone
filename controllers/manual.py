from modes.mode import Mode

# user inputs commands one at a time
class Manual(Mode):

    def __init__(self, drone, log=False, log_path=None):
        super().__init__(drone, log, log_path)

    def run(self):
        # user inputs commands through prompt
        timestep = 0
        while (True):
            cmd = input()
            if cmd == 'quit':
                break
            timestep += 1
            self.command(timestep, cmd)