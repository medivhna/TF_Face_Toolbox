# Copyright 2018 Guanshuo Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.contrib import layers

from resnet import ResNet

class ResNeXt(ResNet):
  def __init__(self, 
               num_layers, 
               num_card=32, 
               weight_decay=0.0005, 
               data_format='NCHW', 
               name='ResNeXt'):
    self.num_card = num_card
    super(ResNeXt, self).__init__(num_layers, 
                                  weight_decay=weight_decay, 
                                  data_format=data_format, 
                                  name=name)

  def resBlock(self, x, 
               num_outputs, 
               stride=1, 
               activation_fn=tf.nn.relu, 
               normalizer_fn=layers.batch_norm, 
               scope=None):
    # Group convolution
    def _group_conv2d(x, num_outputs, num_group, stride=1, scope=''):
      assert num_outputs%num_group==0, "num_outputs must be divided by num_group %d." % num_group
      x_groups = tf.split(x, self.num_card, axis=self.channel_axis)
      y_groups = []
      for idx, x_split in enumerate(x_groups):
        y = layers.conv2d(x_split, num_outputs/num_group, kernel_size=3, stride=stride,
                          scope=scope+'_group_%d'%idx)
        y_groups.append(y)
      x = tf.concat(y_groups, axis=self.channel_axis)

      return x

    assert num_outputs%2==0, "num_outputs must be divided by 2."
    with tf.variable_scope(scope, 'resBlock'):
      shortcut = x
      if stride != 1 or x.get_shape()[self.channel_axis] != num_outputs:
        shortcut = layers.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride, 
                                 activation_fn=None, scope='conv_1x1_shortcut')

      x = layers.conv2d(x, num_outputs/2, kernel_size=1, stride=1, scope='conv1_1x1')
      x = _group_conv2d(x, num_outputs/2, kernel_size=3, num_group=self.num_card, stride=stride, scope='conv2_3x3')
      x = layers.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, scope='conv3_1x1')
      
      x += shortcut
      x = activation_fn(x)

    return x