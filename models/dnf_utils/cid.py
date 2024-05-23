from torch import nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
import mmcv
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp
from torch.nn import functional as F
import torch

def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = nn.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, groups * channels_per_group, height, width)

    return x



class CrossResolutionWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = nn.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = nn.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out

class SpatialWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class ConditionalChannelWeighting(nn.Module):

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):

        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]

            x2 = self.cross_resolution_weighting(x2)

            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

            out = [nn.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
            out = [channel_shuffle(s, 2) for s in out]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

# CI
class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        return self.dconv(x)

# Post-CI
# class MLP(nn.Module):
#     def __init__(self, f_number, excitation_factor=2) -> None:
#         super().__init__()
#         self.act = nn.GELU()
#
#         print("\n")
#         print("\n")
#         print("here pwconv 's  size  is feature_num * in_channels is :" , excitation_factor * f_number)
#         print("f_number is :",f_number)
#         print("\n")
#
#         #self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
#
#         self.pwconv2 = SpatialWeighting(channels=f_number , ratio=4)
#         #self.pwconv3 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)
#
#     def forward(self, x):
#         #x = self.pwconv1(x)
#         #x = self.act(x)
#         x = self.pwconv2(x)
#         #x = self.pwconv3(x)


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.grn = GRN(excitation_factor * f_number)
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv_mid = SpatialWeighting(channels=f_number * excitation_factor, ratio=4)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.act(x)
        x = self.grn(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv_mid(x)
        x = self.pwconv2(x)
        return x + input
    
    
    
class CID(nn.Module):
    def __init__(self, f_number, padding_mode) -> None:
        super().__init__()
        self.channel_independent = DConv7(f_number, padding_mode)
        self.channel_dependent = MLP(f_number, excitation_factor=2)

    def forward(self, x):
        return self.channel_dependent(self.channel_independent(x))
