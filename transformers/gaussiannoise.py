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
					mean = 0,
					deviation = 0.1,
				 ):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		arr = observation.to_numpy()
		noise = np.random.normal(loc=self.mean, scale=self.deviation, size=arr.shape)
		noisy_arr = arr + noise
		if type(observation) == Vector:
			noisy_arr = np.reshape(noisy_arr, (len(noisy_arr),))
		observation.save_transformation(self, noisy_arr)