from transformers.transformer import Transformer
from observations.image import Image
from skimage.transform import resize
import numpy as np
from component import _init_wrapper
from monodepth2 import networks
import torch
from torchvision import transforms
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

# runs MonoDepth2 to extract depths from an RGB FPV-camera
class MonoDepth2(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self):
		super().__init__()
		self.connect_priority = 1

	def connect(self, state=None):
		super().connect(state)
		# fetch monodepth2 model
		self._encoder = networks.ResnetEncoder(18, False)
		self._depth_decoder = networks.DepthDecoder(num_ch_enc=self._encoder.num_ch_enc, scales=range(4))
		loaded_dict_enc = torch.load('local/models/monodepth2/mono_640x192/encoder.pth', map_location='cpu')
		self._encoder.load_state_dict({k: v for k, v in loaded_dict_enc.items() if k in self._encoder.state_dict()})
		self._depth_decoder.load_state_dict(torch.load('local/models/monodepth2/mono_640x192/depth.pth', map_location='cpu'))

	# if observation type is valid, applies transformation
	def transform(self, observation):
		#observation.check(Image)
		img_array = observation.to_numpy()

		
		nRows = 192
		nCols = 640

		with torch.no_grad():
			frame = cv2.resize(img_array, (nCols, nRows))
			input_image = transforms.ToTensor()(frame).unsqueeze(0).float()
			with torch.no_grad():
				features = self._encoder(input_image)
				outputs = self._depth_decoder(features)
			disp = outputs[("disp", 0)]
			disp_np = disp.squeeze().cpu().numpy()
			vmax = np.percentile(disp_np, 50)
			normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
			mapper = cm.ScalarMappable(norm=normalizer, cmap='binary_r')#, cmap='magma')#, cmap='binary_r')
			colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
			depth_img = pil.fromarray(colormapped_im)
			#depth_img.save('depth_img.png')
		
		depth_img = np.array(depth_img)
		depth_img = 255 - depth_img
		# invert all values so that white is far and black is close
		# make channel first for SB3
		depth_img = np.moveaxis(depth_img, 2, 0)
		observation.set_data(depth_img)