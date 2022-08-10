# normalizes dpeth-image observations
from transformers.transformer import Transformer
import numpy as np
import cv2
from observations.image import Image
from component import _init_wrapper

class NormalizeDepth(Transformer):
    # constructor, arbitrary units depend on drone (typically assume meters)
    @_init_wrapper
    def __init__(self, min_depth=0, max_depth=100, name=None):
        super().__init__()

    # if observation type is valid, applies transformation
    def transform(self, observation):
        observation.check(Image)
        image = observation.to_numpy()
        normalized = np.interp(image, (self.min_depth, self.max_depth), (0, 255)).astype('uint8')
        observation.save_transformation(self, normalized)