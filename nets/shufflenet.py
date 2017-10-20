# Copyright 2017 Medivhna. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# ==============================================================================

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

class ShuffleNet(object):
  def __init__(self, num_groups=3, 
               alpha=1, 
               weight_decay=0.0002, 
               data_format='NCHW', 
               name='ShuffleNet'):
    if num_groups == 1:
      self.num_channels = 144
    elif num_groups == 2:
      self.num_channels = 200
    elif num_groups == 3:
      self.num_channels = 240
    elif num_groups == 4:
      self.num_channels = 272
    elif num_groups == 8:
      self.num_channels = 384
    else:
      raise ValueError(
        'Unsupported num_groups')
    self.num_channels = int(alpha*self.num_channels)
    self.alpha = alpha
    self.num_groups = num_groups

    assert data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
    self.data_format = data_format
    if self.data_format == 'NCHW':
      # Version check. Because NCHW data format is not supported in 
      # 'tf.contrib.layers.separable_conv2d' before TensorFlow r1.3
      major, minor, _ = tf.__version__.split('.')
      assert (int(major) == 1 and int(major) >= 3) or int(major) > 1, 'Not supported TensorFlow version. Please update to TensorFlow r1.3 or above.'

    self.weight_decay = weight_decay

    self.name = name

  def resBlock_dw_grouped(self, x, 
                          num_outputs, 
                          num_groups, 
                          stride=1, 
                          activation_fn=tf.nn.relu, 
                          scope=None):
    assert num_outputs%4==0, "num_outputs must be divided by 4."
    assert num_outputs%num_groups==0, "num_outputs must be divided by num_groups %d." % num_groups

    with tf.variable_scope(scope, 'resBlock_dw_grouped'):
      shortcut = x
      if stride != 1:
        shortcut = layers.avg_pool2d(shortcut, 
                                     kernel_size=3, 
                                     stride=stride, 
                                     padding='SAME',
                                     data_format=self.data_format)

      x = self.grouped_conv2d(x, num_outputs/4, 
                              kernel_size=1, 
                              num_groups=self.num_groups)
      x = self.channel_shuffle(x, self.num_groups)
      x = layers.separable_conv2d(x, None, kernel_size=3, 
                                  depth_multiplier=1, 
                                  stride=stride)
      x = self.grouped_conv2d(x, num_outputs, kernel_size=1,
                              num_groups=self.num_groups, 
                              activation_fn=None)
      
      if stride != 1:
        x = tf.concat([x, shortcut], axis=self.channel_axis)
      else:
        x += shortcut

      x = activation_fn(x)

    return x

  def resBlock_dw(self, x, 
                  num_outputs, 
                  stride=1, 
                  activation_fn=tf.nn.relu, 
                  scope=None):
    assert num_outputs%4==0, "num_outputs must be divided by 4."

    with tf.variable_scope(scope, 'resBlock_dw'):
      shortcut = x
      if stride != 1 or x.get_shape()[self.channel_axis] != num_outputs:
        shortcut = layers.conv2d(shortcut, num_outputs, 
                                 kernel_size=1, 
                                 stride=stride, 
                                 activation_fn=None, 
                                 scope='shortcut')

      x = layers.conv2d(x, num_outputs/4, kernel_size=1)
      x = layers.separable_conv2d(x, None, kernel_size=3, 
                                  depth_multiplier=1, 
                                  stride=stride)
      x = layers.conv2d(x, num_outputs, 
                        kernel_size=1, 
                        activation_fn=None)
      
      x += shortcut

      x = activation_fn(x)

    return x

  def grouped_conv2d(self, inputs, num_outputs, kernel_size, num_groups, activation_fn=tf.nn.relu):
    if self.data_format == 'NCHW':
      num_inputs = inputs.get_shape().as_list()[1]
    else:
      num_inputs = inputs.get_shape().as_list()[3]
    assert num_inputs%num_groups==0, "num_inputs %d must be divided by num_groups %d." % (num_inputs, num_groups)

    if self.data_format == 'NCHW':
      layer_input = inputs[:, 0:num_inputs/num_groups, :, :]
    else:
      layer_input = inputs[:, :, :, 0:num_inputs/num_groups]
    outputs = layers.conv2d(layer_input, 
                            num_outputs/num_groups, 
                            kernel_size, 
                            activation_fn=activation_fn)

    for i in xrange(1, num_groups):
      if self.data_format == 'NCHW':
        layer_input = inputs[:, i*num_inputs/num_groups:(i+1)*num_inputs/num_groups, :, :]
      else:
        layer_input = inputs[:, :, :, i*num_inputs/num_groups:(i+1)*num_inputs/num_groups]
      temp = layers.conv2d(inputs[:, :, :, i*num_inputs/num_groups:(i+1)*num_inputs/num_groups], 
                           num_outputs/num_groups, kernel_size, activation_fn=activation_fn)
      outputs = tf.concat([outputs, temp], axis=self.channel_axis)

    return outputs

  def channel_shuffle(self, inputs, num_groups):
    if self.data_format == 'NCHW':
      num_per_group = inputs.get_shape().as_list()[1]
      input_height = inputs.get_shape().as_list()[2]
      input_width = inputs.get_shape().as_list()[3]
    else:
      num_per_group = inputs.get_shape().as_list()[3]
      input_height = inputs.get_shape().as_list()[1]
      input_width = inputs.get_shape().as_list()[2]
    assert num_per_group%num_groups==0, "num_inputs %d must be divided by num_groups %d." % (num_per_group, num_groups)
    num_per_group /= num_groups

    if self.data_format == 'NCHW':
      outputs = tf.reshape([-1, num_groups, num_per_group, input_height, input_width])
      outputs = tf.transpose([0, 2, 1, 3, 4])
      outputs = tf.reshape([-1, num_groups*num_per_group, input_height, input_width])
    else:
      outputs = tf.reshape([-1, input_height, input_width, num_groups, num_per_group])
      outputs = tf.transpose([0, 1, 2, 4, 3])
      outputs = tf.reshape([-1, input_height, input_width, num_groups*num_per_group])

    return outputs


  def __call__(self, inputs, is_training=False, reuse=None):
    with tf.variable_scope(self.name, reuse=reuse):
      with arg_scope([layers.batch_norm], scale=True, fused=True, 
                      data_format=self.data_format, 
                      is_training=is_training):
        with arg_scope([layers.conv2d, layers.separable_conv2d], 
                        activation_fn=tf.nn.relu, 
                        normalizer_fn=layers.batch_norm, 
                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                        data_format=self.data_format):

          if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

          with tf.variable_scope('Stage1'):
            net = layers.conv2d(inputs, num_outputs=int(self.alpha*24), kernel_size=7, stride=2)
            net = layers.max_pool2d(net, kernel_size=3, stride=2, 
                                    padding='SAME', data_format=self.data_format)

          with tf.variable_scope('Stage2'):
            net = self.resBlock_dw(net, num_outputs=self.num_channels, stride=2)
            net = layers.repeat(net, 3, self.resBlock_dw, num_outputs=self.num_channels)

          with tf.variable_scope('Stage3'):
            net = self.resBlock_dw_grouped(net, num_outputs=self.num_channels, 
                                           num_groups=self.num_groups, stride=2)
            net = layers.repeat(net, 7, self.resBlock_dw_grouped, 
                                num_outputs=2*self.num_channels, 
                                num_groups=self.num_groups)

          with tf.variable_scope('Stage4'):
            net = self.resBlock_dw_grouped(net, num_outputs=2*self.num_channels, 
                                           num_groups=self.num_groups, stride=2)
            net = layers.repeat(net, 3, 
                                self.resBlock_dw_grouped, 
                                num_outputs=4*self.num_channels, 
                                num_groups=self.num_groups)

          if self.data_format == 'NCHW':
            net = tf.reduce_mean(net, [2, 3])
            net = tf.reshape(net, [-1, net.get_shape().as_list()[1]])
          else:
            net = tf.reduce_mean(net, [1, 2])
            net = tf.reshape(net, [-1, net.get_shape().as_list()[-1]])

          if is_training:
            net = layers.dropout(net, keep_prob=0.5)  
                   
          pre_logits = layers.fully_connected(features, num_outputs=128, activation_fn=None, 
                               weights_regularizer=layers.l2_regularizer(self.weight_decay))

    return pre_logits

  @property
  def vars(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
