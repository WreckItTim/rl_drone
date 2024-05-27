from torch import nn
from torch import Tensor
from torch.nn import functional as F

# CUSTOM SLIM LAYERS
class SlimMLP(nn.Linear):
    def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True,
                 device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_features, max_out_features, bias, device, dtype)
        self.max_in_features = self.in_features = max_in_features
        self.max_out_features = self.out_features = max_out_features
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_features = max(1, int(self.rho * self.max_in_features))
        if self.slim_out:
            self.out_features = max(1,int(self.rho * self.max_out_features))
        #print(f'B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        y = F.linear(input, weight, bias)
        #utils.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
        return y
class SlimConv2d(nn.Conv2d):
    def __init__(self, max_in_features: int, max_out_features: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_features, max_out_features, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, device=device, dtype=dtype)
        self.max_in_features = self.in_features = max_in_features
        self.max_out_features = self.out_features = max_out_features
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_features = max(1,int(self.rho * self.max_in_features))
        if self.slim_out:
            self.out_features = max(1,int(self.rho * self.max_out_features))
        #print(f'conv2d B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_features,:self.in_features,:,:]
        #print(f'conv2d A4-shape:{weight.shape}')
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        y = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        #utils.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
        return y
class SlimConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, max_in_features: int, max_out_features: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_features, max_out_features, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, device=device, dtype=dtype)
        self.max_in_features = self.in_features = max_in_features
        self.max_out_features = self.out_features = max_out_features
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor, output_size = None) -> Tensor:
        if self.slim_in:
            self.in_features = max(1,int(self.rho * self.max_in_features))
        if self.slim_out:
            self.out_features = max(1,int(self.rho * self.max_out_features))
        #print(f'trans2d B4-shape:{self.weight.shape}')
        weight = self.weight[:self.in_features,:self.out_features,:,:]
        #print(f'trans2d A4-shape:{weight.shape}')
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        num_spatial_dims = 2
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size,
        num_spatial_dims, self.dilation)
        y = F.conv_transpose2d(input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        #utils.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
        return y
class SlimBatchNorm2d(nn.Module):
    def __init__(self, max_features, rhos):
        super().__init__()
        self.max_features = max_features
        self.idx_map = {}
        bns = []
        for idx, rho in enumerate(rhos):
            self.idx_map[rho] = idx
            n_features = max(1,int(max_features*rho))
            bns.append(nn.BatchNorm2d(n_features))
        self.bn = nn.ModuleList(bns)
        self.rho = 1
    def forward(self, input):
        idx = self.idx_map[self.rho]
        y = self.bn[idx](input)
        return y
    
def V1s(rhos):
    return nn.Sequential(
        nn.Sequential(
            SlimConv2d(3, 32, 4, stride=2, padding=1, slim_in=False),
            SlimBatchNorm2d(32, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConv2d(32, 64, 4, stride=2, padding=1),
            SlimBatchNorm2d(64, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConv2d(64, 128, 4, stride=2, padding=1),
            SlimBatchNorm2d(128, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConv2d(128, 256, 4, stride=2, padding=1),
            SlimBatchNorm2d(256, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConv2d(256, 256, 3, padding=2, dilation=2),
            SlimBatchNorm2d(256, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConv2d(256, 256, 3, padding=4, dilation=4),
            SlimBatchNorm2d(256, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConv2d(256, 256, 3, padding=2, dilation=2),
            SlimBatchNorm2d(256, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            SlimBatchNorm2d(128, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            SlimBatchNorm2d(64, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            SlimBatchNorm2d(32, rhos),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            SlimConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            SlimBatchNorm2d(32, rhos),
            nn.SELU(inplace=True),
            SlimConv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            SlimConv2d(32, 1, kernel_size=1, stride=1, padding=0, slim_out=False),
            nn.Sigmoid()
        )
    )

def V1():
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=256), 
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    )

# This is the new model
def ResNet152(bottleneck_channel=512):
    return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, bottleneck_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=bottleneck_channel, num_channels=bottleneck_channel),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(bottleneck_channel, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=128),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=32),
                nn.SELU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
                nn.Sigmoid()
            ),
        )

# CUSTOM SLIM LAYERS
from torch import nn, Tensor
import torch
class SELU_Hack(nn.SELU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)[:,:,:,:(input.shape[3]-1)]
    
class Conv2d_Hack(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(torch.transpose(input, 2, 3))
    
class Sigmoid_Hack(nn.Sigmoid):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(torch.transpose(input, 2, 3))
    
def IanV1_parent():
    return nn.Sequential(
        ############################################ Depth prediction network
        nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=256), 
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ),
    )
    
def IanV1_student(bn_channels):
    return nn.Sequential(
        ############################################ Depth prediction network
        nn.Sequential(
            Conv2d_Hack(3, 32, 4, stride=3),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=3),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(64, bn_channels, 4, stride=3),
            nn.GroupNorm(num_groups=bn_channels, num_channels=bn_channels),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(bn_channels, 64, kernel_size=4, stride=2, padding=(2,0)),
            nn.GroupNorm(num_groups=32, num_channels=64),
            SELU_Hack(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=64, num_channels=128),
            nn.SELU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=2, dilation=2), 
            nn.GroupNorm(num_groups=32, num_channels=256), 
            nn.SELU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            Sigmoid_Hack()
        ),
    )

def get_head_tail(model_name, split_point, compression):
    if model_name == 'V1':
        if split_point == 0:
            head_block = nn.Sequential(
                nn.Conv2d(3, compression, 4, stride=2, padding=1),
                nn.BatchNorm2d(compression),
                nn.SELU(inplace=True),
            )
            tail_block = nn.Sequential(
                nn.Conv2d(compression, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SELU(inplace=True),
            )
        if split_point == 1:
            head_block = nn.Sequential(
                nn.Conv2d(32, compression, 4, stride=2, padding=1),
                nn.BatchNorm2d(compression),
                nn.SELU(inplace=True)
            )
            tail_block = nn.Sequential(
                nn.Conv2d(compression, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128), 
                nn.SELU(inplace=True)
            )
        if split_point == 2:
            head_block = nn.Sequential(
                nn.Conv2d(64, compression, 4, stride=2, padding=1),
                nn.BatchNorm2d(compression),
                nn.SELU(inplace=True)
            )
            tail_block = nn.Sequential(
                nn.Conv2d(compression, 256, 4, stride=2, padding=1), 
                nn.BatchNorm2d(256), 
                nn.SELU(inplace=True)
            )
    if model_name == 'ResNet152':
        if split_point == 4:
            head_block = nn.Sequential(
                nn.Conv2d(512, compression, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=compression, num_channels=compression),
                nn.SELU(inplace=True),
            )
            tail_block = nn.Sequential(
                nn.ConvTranspose2d(compression, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            )
    return head_block, tail_block