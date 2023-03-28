from typing import List

import torch

from src.modules.architectures import aux_modules
from src.utils import common
    

class ConvBasic(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, isBatchNormAfter=True):
        super(ConvBasic, self).__init__()
        if isBatchNormAfter:
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            )

    def forward(self, x):
        x = self.net(x)
        return x


class ConvWithBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, whether_down_sample, isBatchNormAfter, bottleneckFactor):
        '''
        :param whether_down_sample: whether down sample the input
        :param isBNafter: whether BN is after the conv layer
        :param bottleneckFactor: the width of the BN layer
        '''
        super().__init__()
        inner_channels = min(in_channels, bottleneckFactor * out_channels)
        layer = []
        layer.append(ConvBasic(in_channels, inner_channels, kernel_size=1, stride=1,
                               padding=0, bias=False, isBatchNormAfter=isBatchNormAfter))
        stride = 2 if whether_down_sample else 1
        layer.append(ConvBasic(inner_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False, isBatchNormAfter=isBatchNormAfter))    
        self.net = torch.nn.Sequential(*layer)

    def forward(self, x):
        x = self.net(x)
        return x
    

