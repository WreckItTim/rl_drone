# crops image observations
from transformers.transformer import Transformer
from observations.image import Image
from skimage.transform import resize
import numpy as np
from component import _init_wrapper

class CropImage(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self, min_dim1, max_dim1, min_dim2, max_dim2):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		#observation.check(Image)
		img_array = observation.to_numpy()
		cropped = img_array[:, self.min_dim1:self.max_dim1, self.min_dim2:self.max_dim2].copy()
		observation.set_data(cropped)