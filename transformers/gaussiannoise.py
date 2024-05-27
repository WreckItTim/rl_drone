from transformers.transformer import Transformer
import numpy as np
from component import _init_wrapper
from observations.vector import Vector

# adds noise from a normal distribution
class GaussianNoise(Transformer):
	# mean is center of distribution
	# deviation is the scale of distribution (std)
	@_init_wrapper
	def __init__(self, 
					deviation, # standard deviation of gaussian noise
					mean = 0, # deviated around this center
					deviation_amp = 0, # (opt) how much to increase sigma during amp up phase from evaluator
					# these values are stored for amps (do not change) this is for file IO
					original_deviation = None,
				 ):
		super().__init__()
		if original_deviation is None:
			self.original_deviation = deviation

	# reset noise level
	def reset_learning(self, state=None):
		self.deviation = self.original_deviation

	# increase noise level
	def amp_up_noise(self):
		self.deviation += self.deviation_amp

	# if observation type is valid, applies transformation
	def transform(self, observation):
		arr = observation.to_numpy()
		noise = np.random.normal(loc=self.mean, scale=self.deviation, size=arr.shape)
		noisy_arr = arr + noise
		if type(observation) == Vector:
			noisy_arr = np.reshape(noisy_arr, (len(noisy_arr),))
		observation.set_data(noisy_arr)