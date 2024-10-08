from transformers.transformer import Transformer
from component import _init_wrapper
import torch
import numpy as np
import cv2
import pickle

# runs a CNN to transform input
class CNN(Transformer):
	
	# constructor
	@_init_wrapper
	def __init__(self, model_path, device_name, quantize=False, split_point=-1):
		super().__init__()
		self.connect_priority = 1
		self._loaded_models = []

	def connect(self, state=None):
		super().connect(state)
		self._back_to_start = self.model_path
		self.read_model(self.model_path)
		
	def set_slim(self, rho):
		for module in self._model.modules():
			if 'Slim' in str(type(module)):
				module.slim = rho
		
	def read_model(self, model_path):
		# # predictions and evaluations
		#self._loaded_models.append(model_path)
		#pickle.dump(self._loaded_models, open('cnn_loaded_models.p', 'wb'))
		self._model = torch.load(model_path, map_location=self.device_name)
		self.set_slim(1.0)
		#modelB = models.V1s([0.25,0.5,0.75,1.0])
		#modelB.load_state_dict(torch.load(model_path, map_location=self.device_name), strict=False)

	# if observation type is valid, applies transformation
	def transform(self, observation):
		#observation.check(Image)
		img = observation.to_numpy()
		X = np.expand_dims(img, 0)
		
		assumed_mean = np.zeros((3, 1, 1), dtype=np.float32)
		assumed_std = np.zeros((3, 1, 1), dtype=np.float32) 
		assumed_mean[0, 0, 0] = 0.485
		assumed_mean[1, 0, 0] = 0.456
		assumed_mean[2, 0, 0] = 0.406
		assumed_std[0, 0, 0] = 0.229
		assumed_std[1, 0, 0] = 0.224
		assumed_std[2, 0, 0] = 0.225
		X = ((X.astype(np.float32)/255-assumed_mean)/assumed_std)
		X = torch.from_numpy(X).to(self.device_name)

		if self.quantize:
			x_min, x_max = -2, 22
			scale = (x_max - x_min)/255
			for m_idx, module in enumerate(self._model):
				X = module(X)
				if m_idx == self.split_point:
					X = torch.dequantize(torch.quantize_per_tensor(X, scale, x_min, torch.qint8))
			Y = X.detach().cpu().numpy()[0]
		else:
			Y = self._model(X).detach().cpu().numpy()[0]
		
		observation.set_data(Y)
		
	# def start(self, state=None):
	# 	self.read_model(self.model_path)