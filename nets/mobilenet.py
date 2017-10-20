# Copyright 2017 Guanshuo Wang. All Rights Reserved.
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

class MobileNet(object):
  def __init__(self, alpha=1, weight_decay=0.0002, data_format='NCHW', name='MobileNet'):
    self.alpha = alpha

    assert data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
    self.data_format = data_format
    if self.data_format == 'NCHW':
      # Version check. Because NCHW data format is not supported in 
      # 'tf.contrib.layers.separable_conv2d' before TensorFlow r1.3
      major, minor, _ = tf.__version__.split('.')
      assert (int(major) == 1 and int(major) >= 3) or int(major) > 1, 'Not supported TensorFlow version. Please update to TensorFlow r1.3 or above.'

    self.weight_decay = weight_decay

    self.name = name

  def depthwise_separable_conv2d(self, net, num_outputs, kernel_size, stride=1, scope=None):
    with tf.variable_scope(scope, 'depthwise_separable_conv2d'):
      net = layers.separable_conv2d(net, None, kernel_size, depth_multiplier=1, stride=stride)
      net = layers.conv2d(net, num_outputs, kernel_size=1)

    return net

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

          net = layers.conv2d(inputs, num_outputs=int(self.alpha*32), kernel_size=3, stride=2)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*64), kernel_size=3)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*128), kernel_size=3, stride=2)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*128), kernel_size=3)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*256), kernel_size=3, stride=2)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*256), kernel_size=3)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*512), kernel_size=3, stride=2)
          
          #net = layers.repeat(net, 5, self.depthwise_separable_conv2d, num_outputs=int(self.alpha*512), kernel_size=3)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*512), kernel_size=3) 
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*512), kernel_size=3) 
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*512), kernel_size=3) 
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*512), kernel_size=3) 
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*512), kernel_size=3) 

          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*1024), kernel_size=3, stride=2)
          net = self.depthwise_separable_conv2d(net, num_outputs=int(self.alpha*1024), kernel_size=3)

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
