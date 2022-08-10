from modes.mode import Mode

# drone follows static list of commands
class CmdList(Mode):

    def __init__(self, drone, cmd_list, log=False, log_path=None):
        super().__init__(drone, log, log_path)
        self.list = cmd_list

    def run(self):
        for timestep, cmd in enumerate(self.list):
            self.command(timestep + 1, cmd)