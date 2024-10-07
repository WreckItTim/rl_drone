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
				 min_output=0.1, # reserve 0 for missing or erroneous data
				 max_output=1,
				 left = None, # value to set if below min_input (otherwise defaults to min_input)
				 right = None, # value to set if above max_input (otherwise defaults to max_input)
				 ):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		normalized = np.interp(observation.to_numpy(),
						 (self.min_input, self.max_input),
						 (self.min_output, self.max_output),
                       left=self.left,
                       right=self.right,
						 )
		if type(observation) == Vector:
			normalized = np.reshape(normalized, (len(normalized),))
		observation.set_data(normalized)