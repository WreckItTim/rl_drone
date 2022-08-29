# resizes image observations
from transformers.transformer import Transformer
from observations.image import Image
from skimage.transform import resize
from component import _init_wrapper

class ResizeImage(Transformer):
    # constructor
    @_init_wrapper
    def __init__(self, image_shape=(64, 64)):
        super().__init__()

    # if observation type is valid, applies transformation
    def transform(self, observation):
        #observation.check(Image)
        resized = resize(observation.to_numpy(), self.image_shape)
        observation.save_transformation(self, resized)