from transformers.transformer import Transformer
import numpy as np
from component import _init_wrapper

# convert data type
class DataType(Transformer):
	# min/max input will clip input
	# min/max output will MinMax normalize to this range
	@_init_wrapper
	def __init__(self, 
			  	to_type,
				 ):
		super().__init__()

	# if observation type is valid, applies transformation
	def transform(self, observation):
		observation.set_data(observation.to_numpy().astype(self.to_type))