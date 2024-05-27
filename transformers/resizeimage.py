# resizes image observations
from transformers.transformer import Transformer
from observations.image import Image
from skimage.transform import resize
import numpy as np
from component import _init_wrapper

class ResizeImage(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self, image_shape=(84, 84)):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		#observation.check(Image)
		img_array = observation.to_numpy()
		img_array = np.moveaxis(img_array, 0, 2)
		resized = resize(img_array, self.image_shape)
		resized = np.moveaxis(resized, 2, 0)
		observation.set_data(resized)