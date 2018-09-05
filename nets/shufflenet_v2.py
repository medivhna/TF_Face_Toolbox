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

# Reference:
# @inproceedings{shufflenetv2,
#   title={ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design},
#   author={Ningning Ma and Xiangyu Zhang and Hai-Tao Zheng and Jian Sun},
#   booktitle={ECCV},
#   year={2018}
# }

from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from model_parallel import *

import net_base

class ShuffleNet_v2_small(net_base.Network):
  def __init__(self, 
               alpha=1.0, 
               se=False,
               residual=False,
               weight_decay=0.0005, 
               data_format='NCHW', 
               name='ShuffleNet_v2_small'):
    super(ShuffleNet_v2_small, self).__init__(weight_decay, data_format, name)
    if alpha == 0.5:
      self.num_outputs = [24, 48, 96, 1024]
      self.name += '_x0_5'
    elif alpha == 1.0:
      self.num_outputs = [58, 116, 232, 1024]
    elif alpha == 1.5:
      self.num_outputs = [88, 176, 352, 1024]
      self.name += '_x1_5'
    elif alpha == 2.0:
      self.num_outputs = [122, 244, 488, 2048]
      self.name += '_x2'
    self.se = se
    if se:
      self.name += '_se'
    self.residual = residual
    if residual:
      self.name += '_res'

  def _channel_split(self, x):
    num_channels = x.get_shape().as_list()[self.channel_axis]
    split_size = int(0.5*num_channels)
    shortcut, x = tf.split(x, [split_size, num_channels-split_size], axis=self.channel_axis)
    return shortcut, x

  def _channel_shuffle(self, x):
    input_height, input_width = x.get_shape().as_list()[2:4] if self.data_format=='NCHW' else x.get_shape().as_list()[1:3]
    num_channels = x.get_shape().as_list()[self.channel_axis]
    if self.data_format == 'NCHW':
      x = tf.reshape(x, [-1, 2, num_channels/2, input_height, input_width])
      x = tf.transpose(x, [0, 2, 1, 3, 4])
      x = tf.reshape(x, [-1, num_channels, input_height, input_width])
    else:
      x = tf.reshape(x, [-1, input_height, input_width, num_channels/2, 2])
      x = tf.transpose(x, [0, 1, 2, 4, 3])
      x = tf.reshape(x, [-1, input_height, input_width, num_channels])
    return x

  def _squeeze_excitation(self, x):
    num_channels = x.get_shape().as_list()[self.channel_axis]
    w = tf.reduce_mean(x, axis=self.spatial_axis, keepdims=True)
    w = layers.conv2d(w, num_channels/2, kernel_size=1, normalizer_fn=None)
    w = layers.conv2d(w, num_channels, kernel_size=1, normalizer_fn=None, activation_fn=tf.nn.sigmoid)
    x = x*w
    return x

  def separable_resBlock(self, x, 
                         num_outputs, 
                         stride=1, 
                         activation_fn=tf.nn.relu, 
                         normalizer_fn=layers.batch_norm,
                         scope=None):
    residual_flag = self.residual and (stride == 1 and num_outputs == x.get_shape().as_list()[self.channel_axis]) 
    with tf.variable_scope(scope, 'resBlock'):
      # channel_split
      shortcut, x = self._channel_split(x)
      if stride != 1:
        shortcut = layers.separable_conv2d(shortcut, num_outputs, kernel_size=3, stride=stride, 
                                           scope='separable_conv_shortcut_3x3')
        shortcut = layers.conv2d(shortcut, num_outputs, kernel_size=1, stride=1, scope='conv_shortcut_1x1')
      if residual_flag:
        res_shortcut = x
      x = layers.conv2d(x, num_outputs, kernel_size=1, stride=1, scope='conv1_1x1',)
      x = layers.separable_conv2d(x, num_outputs, kernel_size=3, stride=stride, scope='separable_conv2_3x3')
      x = layers.conv2d(x, num_outputs, kernel_size=1, stride=1, scope='conv3_1x1')
      if self.se:
        x = self._squeeze_excitation(x)      
      if residual_flag:
        x += res_shortcut

      # concat
      x = tf.concat([shortcut, x], axis=self.channel_axis)
      x = self._channel_shuffle(x)
      
    return x

  def backbone(self, inputs, is_training=False, reuse=None):
    end_points = OrderedDict()
    with tf.variable_scope(self.name, values=[inputs], reuse=reuse):
      with arg_scope([layers.batch_norm], scale=True, fused=True, 
                      data_format=self.data_format, 
                      is_training=is_training):
        with arg_scope([layers.conv2d], 
                        activation_fn=tf.nn.relu, 
                        normalizer_fn=layers.batch_norm, 
                        biases_initializer=None, 
                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                        data_format=self.data_format):
          with arg_scope([layers.separable_conv2d], 
                          depth_multiplier=1,
                          activation_fn=None, 
                          normalizer_fn=layers.batch_norm, 
                          biases_initializer=None, 
                          weights_regularizer=layers.l2_regularizer(self.weight_decay),
                          data_format=self.data_format):
            if self.data_format == 'NCHW':
              inputs = tf.transpose(inputs, [0, 3, 1, 2])

            with tf.variable_scope('conv1'):
              net = layers.conv2d(inputs, num_outputs=24, kernel_size=3, stride=2, scope='conv_3x3')
              end_points['conv1/conv_3x3'] = net
              net = layers.max_pool2d(net, kernel_size=3, stride=2, padding='SAME', 
                                      data_format=self.data_format, scope='maxpool_3x3_2')
              end_points['conv1/maxpool_3x3_2'] = net

            with tf.variable_scope('conv2'):
              for idx in xrange(4):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[0], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv2/resBlock_%d'%idx] = net

            with tf.variable_scope('conv3'):
              for idx in xrange(8):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[1], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv3/resBlock_%d'%idx] = net

            with tf.variable_scope('conv4'):
              for idx in xrange(4):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[2], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv4/resBlock_%d'%idx] = net

            with tf.variable_scope('conv5'):
              net = layers.conv2d(net, num_outputs=self.num_outputs[3], kernel_size=1, 
                                  stride=1, scope='conv_1x1')
              end_points['conv5/conv_1x1'] = net
                                    
            net = tf.reduce_mean(net, self.spatial_axis)

    return net, end_points

  def forward(self, images, num_classes=None, is_training=True):
    # Forward
    features, end_points = self.backbone(images, is_training=is_training)
    # Logits
    if is_training:
      assert num_classes is not None, 'num_classes must be given when is_training=True'
      with tf.variable_scope('classifier'):
        features_drop = layers.dropout(features, keep_prob=0.5, is_training=is_training)
        logit = layers.fully_connected(features_drop, num_classes, activation_fn=None, 
                                       weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                       biases_initializer=None,
                                       scope='fc_classifier')
      logits = {}
      logits['logits'] = logit
      logits['features'] = features
      return logits
    else:
      # for _, var in end_points.items():
      # 	print(var)
      return features


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


class ShuffleNet_v2_middle(ShuffleNet_v2_small):
  def __init__(self, 
               se=False,
               residual=False,
               weight_decay=0.0005, 
               data_format='NCHW', 
               name='ShuffleNet_v2_middle'):
    super(ShuffleNet_v2_middle, self).__init__(1.0, se, residual, weight_decay, data_format, name)
    self.num_outputs = [244, 488, 976, 1952, 2048]

  def backbone(self, inputs, is_training=False, reuse=None):
    end_points = OrderedDict()
    with tf.variable_scope(self.name, reuse=reuse):
      with arg_scope([layers.batch_norm], scale=True, fused=True, 
                      data_format=self.data_format, 
                      is_training=is_training):
        with arg_scope([layers.conv2d], 
                        activation_fn=tf.nn.relu, 
                        normalizer_fn=layers.batch_norm, 
                        biases_initializer=None, 
                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                        data_format=self.data_format):
          with arg_scope([layers.separable_conv2d], 
                          depth_multiplier=1,
                          activation_fn=None, 
                          normalizer_fn=layers.batch_norm, 
                          biases_initializer=None, 
                          weights_regularizer=layers.l2_regularizer(self.weight_decay),
                          data_format=self.data_format):
            if self.data_format == 'NCHW':
              inputs = tf.transpose(inputs, [0, 3, 1, 2])

            with tf.variable_scope('conv1'):
              net = layers.conv2d(inputs, num_outputs=64, kernel_size=3, stride=2, scope='conv_3x3')
              end_points['conv1/conv_3x3'] = net
              net = layers.max_pool2d(net, kernel_size=3, stride=2, padding='SAME', 
                                      data_format=self.data_format, scope='maxpool_3x3_2')
              end_points['conv1/maxpool_3x3_2'] = net

            with tf.variable_scope('conv2'):
              for idx in xrange(3):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[0], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv2/resBlock_%d'%idx] = net

            with tf.variable_scope('conv3'):
              for idx in xrange(4):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[1], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv3/resBlock_%d'%idx] = net

            with tf.variable_scope('conv4'):
              for idx in xrange(6):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[2], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv4/resBlock_%d'%idx] = net

            with tf.variable_scope('conv5'):
              for idx in xrange(3):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[3], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv4/resBlock_%d'%idx] = net

            with tf.variable_scope('conv5'):
              net = layers.conv2d(net, num_outputs=self.num_outputs[4], kernel_size=1, 
                                  stride=1, scope='conv_1x1')
              end_points['conv6/conv_1x1'] = net
                                    
            net = tf.reduce_mean(net, self.spatial_axis)

    return net, end_points

class ShuffleNet_v2_large(ShuffleNet_v2_small):
  def __init__(self, 
               weight_decay=0.0005, 
               data_format='NCHW', 
               name='ShuffleNet_v2_large'):
    super(ShuffleNet_v2_large, self).__init__(1.0, True, True, weight_decay, data_format, name)
    self.num_outputs = [340, 680, 1360, 2720, 2048]

  def backbone(self, inputs, is_training=False, reuse=None):
    end_points = OrderedDict()
    with tf.variable_scope(self.name, reuse=reuse):
      with arg_scope([layers.batch_norm], scale=True, fused=True, 
                      data_format=self.data_format, 
                      is_training=is_training):
        with arg_scope([layers.conv2d], 
                        activation_fn=tf.nn.relu, 
                        normalizer_fn=layers.batch_norm, 
                        biases_initializer=None, 
                        weights_regularizer=layers.l2_regularizer(self.weight_decay),
                        data_format=self.data_format):
          with arg_scope([layers.separable_conv2d], 
                          depth_multiplier=1,
                          activation_fn=None, 
                          normalizer_fn=layers.batch_norm, 
                          biases_initializer=None, 
                          weights_regularizer=layers.l2_regularizer(self.weight_decay),
                          data_format=self.data_format):
            if self.data_format == 'NCHW':
              inputs = tf.transpose(inputs, [0, 3, 1, 2])

            with tf.variable_scope('conv1'):
              net = layers.conv2d(inputs, num_outputs=64, kernel_size=3, stride=2, scope='conv1_3x3')
              end_points['conv1/conv1_3x3'] = net
              net = layers.conv2d(inputs, num_outputs=64, kernel_size=3, scope='conv2_3x3')
              end_points['conv1/conv2_3x3'] = net
              net = layers.conv2d(inputs, num_outputs=128, kernel_size=3, scope='conv3_3x3')
              end_points['conv1/conv3_3x3'] = net
              net = layers.max_pool2d(net, kernel_size=3, stride=2, padding='SAME', 
                                      data_format=self.data_format, scope='maxpool_3x3_2')
              end_points['conv1/maxpool_3x3_2'] = net

            with tf.variable_scope('conv2'):
              for idx in xrange(10):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[0], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv2/resBlock_%d'%idx] = net

            with tf.variable_scope('conv3'):
              for idx in xrange(10):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[1], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv3/resBlock_%d'%idx] = net

            with tf.variable_scope('conv4'):
              for idx in xrange(23):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[2], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv4/resBlock_%d'%idx] = net

            with tf.variable_scope('conv5'):
              for idx in xrange(10):
                net = self.separable_resBlock(net, num_outputs=self.num_outputs[3], 
                                    stride=2 if not idx else 1, scope='resBlock_%d'%idx)
                end_points['conv4/resBlock_%d'%idx] = net

            with tf.variable_scope('conv5'):
              net = layers.conv2d(net, num_outputs=self.num_outputs[4], kernel_size=1, 
                                  stride=1, scope='conv_1x1')
              end_points['conv6/conv_1x1'] = net
                                    
            net = tf.reduce_mean(net, self.spatial_axis)

    return net, end_points
