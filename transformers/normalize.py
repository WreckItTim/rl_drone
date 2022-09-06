from transformers.transformer import Transformer
import numpy as np
from component import _init_wrapper
from observations.vector import Vector

# MinMax normalizes observations
class Normalize(Transformer):
	# min/max input will clip input
	# min/max output will MinMax normalize to this range
	@_init_wrapper
	def __init__(self, 
				 min_input=0, 
				 max_input=255,
				 min_output=0, 
				 max_output=100,
				 ):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		normalized = np.interp(observation.to_numpy(),
						 (self.min_input, self.max_input),
						 (self.min_output, self.max_output),
						 )
		if type(observation) == Vector:
			normalized = np.reshape(normalized, (len(normalized),))
			for idx, name in enumerate(observation.names):
				observation.names[idx] = observation.names[idx] + '_normalized'
		observation.save_transformation(self, normalized)