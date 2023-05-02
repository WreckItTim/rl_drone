
# CUSTOM SLIM LAYERS
class Slim(nn.Linear):
	def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True,
				 device=None, dtype=None,
				slim_in=True, slim_out=True) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super().__init__(max_in_features, max_out_features, bias, device, dtype)
		self.max_in_features = max_in_features
		self.max_out_features = max_out_features
		self.slim_in = slim_in
		self.slim_out = slim_out
		self.slim = 1
		
	def forward(self, input: Tensor) -> Tensor:
		if self.slim_in:
			self.in_features = int(self.slim * self.max_in_features)
		if self.slim_out:
			self.out_features = int(self.slim * self.max_out_features)
		#print(f'B4-shape:{self.weight.shape}')
		weight = self.weight[:self.out_features, :self.in_features]
		if self.bias is not None:
			bias = self.bias[:self.out_features]
		else:
			bias = self.bias
		y = F.linear(input, weight, bias)
		#utils.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
		return y
        
def convert_to_slim(model):
	#after calling, set as such: new_model = copy.deepcopy(model) ..
	nLinearLayers = 0
	for module in model.modules():
		if 'Linear' in str(type(module)):
			nLinearLayers += 1
	modules = []
	onLinear = 0
	for module in model.modules():
		if 'Sequential' in str(type(module)):
			continue
		elif 'Linear' in str(type(module)):
			onLinear += 1
			max_in_features = module.in_features
			max_out_features = module.out_features
			bias = module.bias is not None
			slim_in, slim_out = True, True
			if onLinear == 1:
				slim_in = False
			if onLinear == nLinearLayers:
				slim_out = False
			new_module = Slim(max_in_features, max_out_features,
							bias=bias, slim_in=slim_in, slim_out=slim_out)
			modules.append(new_module)
		else:
			modules.append(module)
	new_model = nn.Sequential(*modules)
	new_model.load_state_dict(copy.deepcopy(model.state_dict()))
	return new_model