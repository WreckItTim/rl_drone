from modes.mode import Mode

# drone follows static path
class Path(Mode):

    def __init__(self, drone, path, moveTo, log=False, log_path=None):
        super().__init__(drone, log, log_path)
        self.path = path

    def run(self):
        for timestep, point in enumerate(self.path):
            if moveTo:
                cmd = utils.move_to_string(point)
            else:
                cmd = utils.moveTo_to_string(point)
            self.command(timestep+1, cmd)