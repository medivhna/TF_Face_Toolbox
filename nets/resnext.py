# Copyright 2017 Medivhna. All Rights Reserved.
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

class ResNeXt(object):
  def __init__(self, num_layers=50, 
               num_card=32, 
               weight_decay=0.0005, 
               data_format='NCHW', 
               name='ResNeXt'):
    assert (num_layers-2)%3==0, "num_layers-2 must be divided by 3."
    self.num_layers = num_layers
    self.num_card = num_card

    self.channel_factor = 2 if num_card > 1 else 4
    if self.num_layers in [50, 101]:
      self.num_block=[3, 4, (self.num_layers-32)/3, 3]
    elif self.num_layers == 152:
      self.num_block=[3, 8, 36, 3]
    elif self.num_layers == 26:
      self.num_block=[2, 2, 2, 2]
    else:
      raise ValueError('Unsupported num_layers.')
    self.num_outputs=[256, 512, 1024, 2048]

    assert data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
    self.data_format = data_format
    self.weight_decay = weight_decay

    self.name = name+'-'+str(num_layers)

  def resBlock(self, x, 
               num_outputs, 
               stride=1, 
               activation_fn=tf.nn.relu, 
               normalizer_fn=layers.batch_norm, 
               scope=None):
    assert num_outputs%self.channel_factor==0, "num_outputs must be divided by channel_factor %d." % self.channel_factor
    assert num_outputs%self.num_card==0, "num_outputs must be divided by num_card %d." % self.num_card

    # Data format
    with tf.variable_scope(scope, 'resBlock'):
      shortcut = x
      if stride != 1 or x.get_shape()[3] != num_outputs:
        shortcut = layers.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride, 
                              activation_fn=None, normalizer_fn=None, scope='shortcut')

      x = layers.conv2d(x, num_outputs/self.channel_factor, kernel_size=1, stride=1)
      x = layers.conv2d(x, num_outputs/self.channel_factor, kernel_size=3, stride=stride)
      x = layers.conv2d(x, num_outputs, kernel_size=1, stride=1, 
                        activation_fn=None, normalizer_fn=None)
      
      x += shortcut

      x = normalizer_fn(x)
      x = activation_fn(x)

    return x

  def __call__(self, inputs, is_training=False, reuse=None):
    with tf.variable_scope(self.name, reuse=reuse):
      with arg_scope([layers.batch_norm], scale=True, fused=True, 
                      data_format=self.data_format, 
                      is_training=is_training):
        with arg_scope([layers.conv2d], activation_fn=tf.nn.relu, 
                        normalizer_fn=layers.batch_norm, 
                        biases_initializer=None, 
                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                        data_format=self.data_format):

          if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

          with tf.variable_scope('conv1'):
            net = layers.conv2d(inputs, num_outputs=64, kernel_size=7, stride=2)
            net = layers.max_pool2d(net, kernel_size=3, stride=2, 
                                    padding='SAME', data_format=self.data_format)

          with tf.variable_scope('conv2'):
            net = layers.repeat(net, self.num_block[0], self.resBlock, self.num_outputs[0])

          with tf.variable_scope('conv3'):
            net = self.resBlock(net, num_outputs=self.num_outputs[1], stride=2)
            net = layers.repeat(net, self.num_block[1]-1, self.resBlock, self.num_outputs[1])

          with tf.variable_scope('conv4'):
            net = self.resBlock(net, num_outputs=self.num_outputs[2], stride=2)
            net = layers.repeat(net, self.num_block[2]-1, self.resBlock, self.num_outputs[2])

          with tf.variable_scope('conv5'):
            net = self.resBlock(net, num_outputs=self.num_outputs[3], stride=2)
            net = layers.repeat(net, self.num_block[3]-1, self.resBlock, self.num_outputs[3])

          if self.data_format == 'NCHW':
            net = tf.reduce_mean(net, [2, 3])
            net = tf.reshape(net, [-1, net.get_shape().as_list()[1]])
          else:
            net = tf.reduce_mean(net, [1, 2])
            net = tf.reshape(net, [-1, net.get_shape().as_list()[-1]])

          if is_training:
            net = layers.dropout(net, keep_prob=0.5)

          pre_logits = layers.fully_connected(net, num_outputs=128, activation_fn=None, 
                          weights_regularizer=layers.l2_regularizer(self.weight_decay))

    return pre_logits

  @property
  def variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)