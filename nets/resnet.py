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

from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

import net_base

class ResNet(net_base.Network):
  def __init__(self, 
               num_layers, 
               pre_act=False,
               weight_decay=0.0005, 
               data_format='NCHW', 
               name='ResNet'):
    assert (num_layers-2)%3==0, "num_layers-2 must be divided by 3."
    self.num_layers = num_layers
    self.pre_act = pre_act

    if self.num_layers in [50, 101]:
      self.num_block=[3, 4, (self.num_layers-32)/3, 3]
    elif self.num_layers == 152:
      self.num_block=[3, 8, 36, 3]
    elif self.num_layers == 26:
      self.num_block=[2, 2, 2, 2]
    else:
      raise ValueError('Unsupported num_layers.')
    self.num_outputs=[256, 512, 1024, 2048]

    super(ResNet, self).__init__(weight_decay, data_format, name+'-'+str(num_layers))

  def conv_bn_relu(self, x, 
                   num_outputs, 
                   kernel_size,
                   stride=1, 
                   activation_fn=tf.nn.relu, 
                   normalizer_fn=layers.batch_norm,
                   scope=None):
    net = layers.conv2d(x, num_outputs, kernel_size=kernel_size, stride=stride, scope=scope)
    tf.add_to_collection('conv_output', net)
    if normalizer_fn is not None:
      net = normalizer_fn(net, scope=scope+'/BatchNorm')
    if activation_fn is not None:
      net = activation_fn(net)

    return net

  def resBlock(self, x, 
               num_outputs, 
               stride=1, 
               activation_fn=tf.nn.relu, 
               normalizer_fn=layers.batch_norm,
               scope=None):
    with tf.variable_scope(scope, 'resBlock'):
      shortcut = x
      if stride != 1 or x.get_shape()[self.channel_axis] != num_outputs:
        if self.pre_act:
          shortcut = layers.batch_norm(shortcut)
        shortcut = self.conv_bn_relu(shortcut, num_outputs, kernel_size=1, stride=stride, 
                                 normalizer_fn=layers.batch_norm if not self.pre_act else None, 
                                 activation_fn=None, 
                                 scope='conv_shortcut_1x1')
        
      if self.pre_act:
        x = tf.nn.relu(batch_norm(x))
      x = self.conv_bn_relu(x, num_outputs/4, kernel_size=1, stride=1, scope='conv1_1x1',)
      x = self.conv_bn_relu(x, num_outputs/4, kernel_size=3, stride=stride, scope='conv2_3x3')
      x = self.conv_bn_relu(x, num_outputs, kernel_size=1, stride=1,  
                        normalizer_fn=normalizer_fn if not self.pre_act else None,
                        activation_fn=None,
                        scope='conv3_1x1')
      
      x += shortcut
      if not self.pre_act:
        x = activation_fn(x)

    return x

  def backbone(self, inputs, is_training=False, reuse=None):
    end_points = OrderedDict()
    with tf.variable_scope(self.name, reuse=reuse):
      with arg_scope([layers.batch_norm], scale=True, fused=True, 
                      data_format=self.data_format, 
                      is_training=is_training):
        with arg_scope([layers.conv2d], 
                        activation_fn=None, 
                        normalizer_fn=None, 
                        biases_initializer=None, 
                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                        data_format=self.data_format):
          if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

          with tf.variable_scope('conv1'):
            net = self.conv_bn_relu(inputs, num_outputs=64, kernel_size=7, stride=2, 
                                activation_fn=tf.nn.relu if not self.pre_act else None,
                                normalizer_fn=layers.batch_norm if not self.pre_act else None,
                                scope='conv_7x7')
            end_points['conv1/conv_7x7'] = net
            net = layers.max_pool2d(net, kernel_size=3, stride=2, 
                                    padding='SAME', data_format=self.data_format, scope='maxpool_3x3_2')
            end_points['conv1/maxpool_3x3_2'] = net

          with tf.variable_scope('conv2'):
            for idx in xrange(self.num_block[0]):
              net = self.resBlock(net, num_outputs=self.num_outputs[0], scope='resBlock_%d'%idx)
              end_points['conv2/resBlock_%d'%idx] = net

          with tf.variable_scope('conv3'):
            for idx in xrange(self.num_block[1]):
              net = self.resBlock(net, num_outputs=self.num_outputs[1], 
                                  stride=2 if not idx else 1, scope='resBlock_%d'%idx)
              end_points['conv3/resBlock_%d'%idx] = net

          with tf.variable_scope('conv4'):
            for idx in xrange(self.num_block[2]):
              net = self.resBlock(net, num_outputs=self.num_outputs[2], 
                                  stride=2 if not idx else 1, scope='resBlock_%d'%idx)
              end_points['conv4/resBlock_%d'%idx] = net

          with tf.variable_scope('conv5'):
            for idx in xrange(self.num_block[3]):
              net = self.resBlock(net, num_outputs=self.num_outputs[3], 
                                  stride=2 if not idx else 1, scope='resBlock_%d'%idx)
              end_points['conv5/resBlock_%d'%idx] = net
                                  
          net = tf.reduce_mean(net, self.spatial_axis)

    return net, end_points

  def forward(self, images, num_classes=None, is_training=True):
    assert num_classes is not None, 'num_classes must be given when is_training=True'
    # Forward
    features, _ = self.backbone(images, is_training=is_training)
    # Logits
    with tf.variable_scope('classifier'):
      features_drop = layers.dropout(features, keep_prob=0.5, is_training=is_training)
      logit = layers.fully_connected(features_drop, num_classes, activation_fn=None, 
                                     weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                     weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                     biases_initializer=None,
                                     scope='fc_classifier')
    logits = {}
    logits['logits'] = logit

    return logits


  def loss_function(self, scope, labels, **logits):
    losses = []
    losses_name = []
    others = {}
    cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits['logits'], 
                                                                labels=labels, 
                                                                scope='cross_entropy')
    losses.append(cross_entropy_loss)
    losses_name.append('cross_entropy')

    # Regularization
    losses, losses_name = self._regularize(scope, losses, losses_name)

    return losses, losses_name, others

  def param_list(self, is_training, trainable, scope=None):
    var_fn = tf.trainable_variables if trainable else tf.global_variables
    scope_name = scope.name+'/' if scope is not None else ''
    if is_training:
      return [var_fn(scope_name+self.name), var_fn(scope_name+'classifier')]
    else:
      return [var_fn(scope_name+self.name)]
    
  def pretrained_param(self, scope=None):
    pretrained_param = []
    for param in self.param_list(is_training=False, trainable=False, scope=scope):
      for v in param:
        if self.name in v.name:
          pretrained_param.append(v)
    return pretrained_param
