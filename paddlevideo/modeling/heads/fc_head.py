# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.nn import AdaptiveAvgPool2D, Linear, Dropout

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class FCHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cfg=dict(name='CrossEntropyLoss'),
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cfg, **kwargs)

        self.fc = Linear(in_channels, num_classes)

    def forward(self, x, seg_num):
        x = self.fc(x)
        return x
    
