from transformers.transformer import Transformer
import numpy as np
from component import _init_wrapper
from observations.vector import Vector
from skimage.filters import gaussian

# Gaussian blurs an image
class GaussianBlur(Transformer):
	# sigma is scale (std) of blur
	@_init_wrapper
	def __init__(self, 
					sigma,
					sigma_amp = 0, # (opt) how much to increase sigma during amp up phase from evaluator
				 # these values are stored for amps (do not change) this is for file IO
				 original_sigma = None,
				 ):
		super().__init__()
		if original_sigma is None:
			self.original_sigma = sigma

	# reset noise level
	def reset_learning(self, state=None):
		self.sigma = self.original_sigma

	# increase noise level
	def amp_up_noise(self):
		self.sigma += self.sigma_amp


	# if observation type is valid, applies transformation
	def transform(self, observation):
		img_array = observation.to_numpy()

		blurred = gaussian(img_array,
			sigma=self.sigma,
			channel_axis=0,
		)

		observation.set_data(blurred)