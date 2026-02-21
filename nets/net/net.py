import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class Unit2D(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 kernel_size,
                 stride=1,
                 dim=2,
                 dropout=0,
                 multiscale=False,
                 bias=True):
        super(Unit2D, self).__init__()
        pad = int((kernel_size - 1) / 2)
        #print("Pad Temporal ", pad)
        if multiscale:
            self.conv1 = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size-2, 1),
                padding=(int((kernel_size - 3) / 2), 0),
                stride=(stride, 1),
                bias=bias)
            self.conv2 = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size+4, 1),
                padding=(int((kernel_size +3) / 2), 0),
                stride=(stride, 1),
                bias=bias)
            self.conv3 = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size+10, 1),
                padding=(int((kernel_size +9) / 2), 0),
                stride=(stride, 1),
                bias=bias)
            self.conv4 = nn.Conv2d(
                D_out*3,
                D_out,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
                bias=bias)
        else:
            if dim == 2:
                self.conv = nn.Conv2d(
                    D_in,
                    D_out,
                    kernel_size=(kernel_size, 1),
                    padding=(pad, 0),
                    stride=(stride, 1),
                    bias=bias)
            elif dim == 3:
                #print("Pad Temporal ", pad)
                self.conv = nn.Conv2d(
                    D_in,
                    D_out,
                    kernel_size=(1, kernel_size),
                    padding=(0, pad),
                    stride=(1, stride),
                    bias=bias)
            else:
                raise ValueError()
     
        self.bn = nn.BatchNorm2d(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.multiscale = multiscale

        # initialize
        
        if multiscale:
           conv_init(self.conv1)
           conv_init(self.conv2)
           conv_init(self.conv3)
           conv_init(self.conv4)
        else:
           conv_init(self.conv)

    def forward(self, x):
        x = self.dropout(x)
        if self.multiscale:
           x = torch.cat((self.bn(self.conv1(x)),self.bn(self.conv2(x)),self.bn(self.conv3(x))),dim=1)
           x = self.relu(self.bn(self.conv4(x)))
        else:
           x = self.relu(self.bn(self.conv(x)))
        return x


def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n = n*k
    module.weight.data.normal_(0, math.sqrt(2. / n))


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
