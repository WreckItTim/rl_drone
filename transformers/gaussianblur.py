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
				 ):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		img_array = observation.to_numpy()

		blurred = gaussian(img_array,
			sigma=self.sigma,
			channel_axis=0,
		)

		observation.set_data(blurred)