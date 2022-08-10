from modes.mode import Mode

# drone will train an RL model
class RLEvaluate(Mode):

    def __init__(self, model, log=False, log_path=None):
        super().__init__(None, log, log_path)
        self.model = model

    def run(self):
        #self.model.evaluate(...)
        pass