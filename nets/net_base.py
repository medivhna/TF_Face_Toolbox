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

import re
import abc
import six

import tensorflow as tf

def net_select(name, data_format='NCHW', weight_decay=5e-4):
  if name == 'SphereNet':
    from sphere import SphereNet
    network = SphereNet(data_format=data_format, 
                        weight_decay=weight_decay)  
  elif name == 'ResNeXt-26':
    from resnext import ResNeXt
    network = ResNeXt(num_layers=26, num_card=32, 
                      data_format=data_format, 
                      weight_decay=weight_decay)
  elif name == 'ResNet-50':
    from resnet import ResNet
    network = ResNet(num_layers=50, 
                     data_format=data_format, 
                     weight_decay=weight_decay)
  elif name == 'ShuffleNet-v2-small':
    from shufflenet_v2 import ShuffleNet_v2_small
    network = ShuffleNet_v2_small(alpha=2.0, 
                                  se=False, residual=False,
                                  data_format=data_format, 
                                  weight_decay=weight_decay)
  elif name == 'ShuffleNet-v2-middle':
    from shufflenet_v2 import ShuffleNet_v2_middle
    network = ShuffleNet_v2_middle(se=False, residual=False,
                                   data_format=data_format, 
                                   weight_decay=weight_decay)
  elif name == 'ShuffleNet-v2-large':
    from shufflenet_v2 import ShuffleNet_v2_large
    network = ShuffleNet_v2_large(data_format=data_format, 
                                  weight_decay=weight_decay)
  elif name == 'MobileNet-v2':
    pass
  elif name == 'Inception-v4':
    pass
  elif name == 'VGG16':
    pass
  elif name == 'AlexNet':
    pass
  else:
    raise ValueError('Unsupport network architecture.')

  return network

@six.add_metaclass(abc.ABCMeta)
class Network(object):
  def __init__(self, 
               weight_decay,  
               data_format, 
               name=None):

    assert data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
    self.data_format = data_format
    self.channel_axis = 1 if self.data_format == 'NCHW' else 3
    self.spatial_axis = [2, 3] if self.data_format == 'NCHW' else [1, 2]

    self.weight_decay = weight_decay
    self.name = name

  @abc.abstractmethod
  def backbone(self, inputs, is_training, reuse):
    pass

  @abc.abstractmethod
  def forward(self, images, num_classes, is_training):
    pass

  @abc.abstractmethod
  def loss_function(self, scope, labels, **logits):
    pass

  def param_list(self, is_training, trainable, scope=None):
    var_fn = tf.trainable_variables if trainable else tf.global_variables
    scope_name = scope.name+'/' if scope is not None else ''
    return [var_fn(scope_name+'/'+self.name)]

  def mult_lr_list(self, scope=None):
    mult_lr_list = []
    for _ in range(len(self.param_list(is_training=True, trainable=True, scope=scope))):
      mult_lr_list.append(1.0)
    return mult_lr_list

  def _regularize(self, scope, losses, losses_name):
    # regularization_loss
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope), name='reg_loss')
    losses.append(regularization_loss)
    losses_name.append('reg_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses)
    for l in losses:
      loss_name = re.sub('TOWER_[0-9]*/', '', l.op.name)
      tf.summary.scalar(loss_name +' (raw)', l)
      tf.summary.scalar(loss_name, loss_averages.average(l))

    return losses, losses_name