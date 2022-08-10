from modes.mode import Mode

# drone will train an RL model
class RLTrain(Mode):

    def __init__(self, model, log=False, log_path=None):
        super().__init__(None, log, log_path)
        self.model = model

    def run(self):

        self.model.learn(total_timesteps=1e4))

        self.model.save(log_path + 'model')