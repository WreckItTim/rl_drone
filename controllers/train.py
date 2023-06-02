from controllers.controller import Controller
from component import _init_wrapper

# trains a reinforcment learning algorithm
# launches learn() from the mdoel object
	# which links to the train environment
	# learn() calls step() and start()
class Train(Controller):
	# constructor
	@_init_wrapper
	def __init__(self, 
			model_component = 'Model',
			train_environment_component = 'TrainEnvironment',
			continue_training = True,
			max_episodes = 10_000,
			random_start = 40, # number of episodes to randomize actions
			train_start = 40, # don't call train() until after train_start episodes
			train_freq = 1, # then call train() every train_freq episode
			batch_size = 100, # split training into mini-batches of steps from buffer
			num_batches = -1, # split training into mini-batches of steps from buffer
			with_distillation = False, # slims during train() and distills to output of super
			use_wandb = True, # turns on logging to wandb
			project_name = 'void', # project name in wandb
		):
		super().__init__()

	# runs control on components
	def run(self):
		# this will continually train model from same learning loop
		# can be used to checkpoint (pause and continue later)
		# can be used for transfer./
		if not self.continue_training:
			self._configuration.reset_all()
		# learn baby learn
		self._model.learn(
			train_environment = self._train_environment,
			max_episodes = self.max_episodes,
			random_start = self.random_start,
			train_start = self.train_start, # don't call train() until after train_start episodes
			train_freq = self.train_freq, # then call train() every train_freq episode
			batch_size = self.batch_size, # split training into mini-batches of steps from buffer
			num_batches = self.num_batches,
			with_distillation = self.with_distillation, # slims during train() and distills to output of super
			use_wandb = self.use_wandb, # turns on logging to wandb
			project_name = self.project_name, # project name in wandb
		)