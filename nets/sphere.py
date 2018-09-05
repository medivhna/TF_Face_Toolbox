# Copyright 2017 Guanshuo Wang. All Rights Reserved.
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
from tensorflow.contrib.framework import arg_scope
from collections import OrderedDict

import net_base

class SphereNet(net_base.Network):
  def __init__(self, weight_decay=0.0005, data_format='NCHW', name='SphereNet'):
    super(SphereNet, self).__init__(weight_decay, data_format, name)
    self.num_outputs=[64, 128, 256, 512]


  def prelu(self, x, name='prelu'):
    shape = [1, 1, 1, 1]
    shape[self.channel_axis] = x.get_shape().as_list()[self.channel_axis]

    alpha = tf.get_variable('alpha', shape,
                            initializer=tf.constant_initializer(0.25),
                            dtype=tf.float32)
    return tf.nn.relu(x) + alpha*(x-tf.abs(x))*0.5

  def resBlock(self, x, num_outputs, scope=None):
    with tf.variable_scope(scope, 'resBlock'):
      shortcut = x
      x = layers.conv2d(x, num_outputs, kernel_size=3, biases_initializer=None, weights_initializer=tf.random_normal_initializer(0.0, 0.01))      
      x = layers.conv2d(x, num_outputs, kernel_size=3, biases_initializer=None, weights_initializer=tf.random_normal_initializer(0.0, 0.01))
      x += shortcut

    return x

  def backbone(self, inputs, is_training=False, reuse=None):
    with tf.variable_scope(self.name, reuse=reuse):
        with arg_scope([layers.conv2d], activation_fn=self.prelu, 
                                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                        data_format=self.data_format):

          if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

          with tf.variable_scope('conv1'):
            net = layers.conv2d(inputs, num_outputs=self.num_outputs[0], kernel_size=3, stride=2) # 64*64*64
            net = self.resBlock(net, num_outputs=self.num_outputs[0]) # 64*64*64

          with tf.variable_scope('conv2'):
            net = layers.conv2d(net, num_outputs=self.num_outputs[1], kernel_size=3, stride=2) # 32*32*128
            net = layers.repeat(net, 2, self.resBlock, self.num_outputs[1]) # 32*32*128 x2

          with tf.variable_scope('conv3'):
            net = layers.conv2d(net, num_outputs=self.num_outputs[2], kernel_size=3, stride=2) # 16*16*256
            net = layers.repeat(net, 4, self.resBlock, self.num_outputs[2]) # 16*16*256 x4

          with tf.variable_scope('conv4'):
            net = layers.conv2d(net, num_outputs=self.num_outputs[3], kernel_size=3, stride=2) # 8*8*512
            net = self.resBlock(net, num_outputs=self.num_outputs[3]) # 8*8*512

          net = tf.reshape(net, [-1, net.get_shape().as_list()[1]*net.get_shape().as_list()[2]*net.get_shape().as_list()[3]])
          net = layers.fully_connected(net, num_outputs=512, activation_fn=None, 
                                       weights_regularizer=layers.l2_regularizer(self.weight_decay)) # 512

    return net

  def forward(self, images, num_classes=None, is_training=True):
    if is_training:
      assert num_classes is not None, 'num_classes must be given when is_training=True'
      # Forward
      features = self.backbone(images, is_training=is_training)
      # Logits
      with tf.variable_scope('classifier'):
        print(features)
        logit = layers.fully_connected(features, num_classes, activation_fn=None, 
                                       weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                       biases_initializer=None,
                                       scope='fc_classifier')
      print(num_classes)
      logits = {}
      logits['logits'] = logit

      return logits
    else:
      features = self.backbone(images, is_training=is_training)
      features_flipped = self.backbone(tf.reverse(images, axis=[2]), is_training=is_training, reuse=True)
      features = (features+features_flipped)/2

      return features

  def loss_function(self, scope, labels, **logits):
    losses = []
    losses_name = []
    others = OrderedDict()
    
    outputs = logits['logits']
    cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(logits=outputs, 
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
