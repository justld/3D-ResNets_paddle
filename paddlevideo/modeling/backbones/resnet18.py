# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from paddle.nn import (Conv2D, BatchNorm2D, Linear, Dropout, MaxPool2D,
                       AvgPool2D)
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from ..registry import BACKBONES
from ..weight_init import weight_init_
from ...utils import load_ckpt


class ConvBNRelu(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv3D(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        self.bn = nn.BatchNorm3D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNRelu(in_channels, out_channels, 3, stride=stride, padding='same'),
            nn.Conv3D(out_channels, out_channels, 3, stride=1, padding='same'),
            nn.BatchNorm3D(out_channels),
        )

        if in_channels != out_channels or stride != 2:
            self.downsample = nn.Sequential(
                nn.Conv3D(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3D(out_channels),
            )
        else:
            self.downsample = Identity()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        res = self.downsample(x)
        x = self.block(x)
        x = self.relu(x + res)
        return x


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


@BACKBONES.register()
class ResNet18(nn.Layer):
    def __init__(self, in_channels=3, num_seg=8):
        super(ResNet18, self).__init__()
        self.num_seg = num_seg

        self.stem = nn.Sequential(
            ConvBNRelu(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias_attr=False),
            nn.MaxPool3D(kernel_size=(3, 3, 3), stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)

        self.pool = nn.AdaptiveAvgPool3D(1)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1):
        layer_list = []
        layer_list.append(Block(in_channels, out_channels, stride))
        for i in range(1, n_blocks):
            layer_list.append(Block(out_channels, out_channels))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.reshape([-1, self.num_seg, c, h, w]).transpose([0, 2, 1, 3, 4]) # N, C, T, H, W
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # N, C, T, H, W
        x = self.pool(x).reshape([paddle.shape(x)[0], -1]) # N, C
        return x








        