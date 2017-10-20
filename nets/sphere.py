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

class SphereFace(object):
  def __init__(self, weight_decay=0.0005, data_format='NCHW', name='SphereFace'):
    self.num_outputs=[64, 128, 256, 512]

    assert data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
    self.data_format = data_format
    self.weight_decay = weight_decay

    self.name = name

  def prelu(self, x, name='prelu'):
    channel_axis = 1 if self.data_format == 'NCHW' or 3
    shape = [1, 1, 1, 1]
    shape[channel_axis] = x.get_shape().as_list()[channel_axis]

    alpha = tf.get_variable('alpha', shape,
                            initializer=tf.constant_initializer(0.25),
                            dtype=tf.float32)
    return tf.nn.relu(x) + alpha*(x-abs(x))*0.5

  def resBlock(self, x, num_outputs, scope=None):
    with tf.variable_scope(scope, 'resBlock'):
      shortcut = x
      x = layers.conv2d(x, num_outputs, kernel_size=3, 
                       weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                       biases_initializer=None)      
      x = layers.conv2d(x, num_outputs, kernel_size=3, 
                       weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                       biases_initializer=None)
      x += shortcut

    return x

  def __call__(self, inputs, reuse=None):
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

          if self.data_format == 'NCHW':
            net = tf.reduce_mean(net, [2, 3])
            net = tf.reshape(net, [-1, net.get_shape().as_list()[1]])
          else:
            net = tf.reduce_mean(net, [1, 2])
            net = tf.reshape(net, [-1, net.get_shape().as_list()[-1]])
          net = layers.fully_connected(net, num_outputs=512, activation_fn=None, 
                    weights_regularizer=layers.l2_regularizer(self.weight_decay)) # 512

    return net

  @property
  def vars(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
