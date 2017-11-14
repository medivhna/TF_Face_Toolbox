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

import math
import tensorflow as tf
from tensorflow.python.layers import base

class Margin_Dense(base.Layer):
    def __init__(self, units, 
                 m, min_lambda, base, gamma, power,
                 activation=None,
                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                 kernel_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Margin_Dense, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.m = m
        self.min_lambda = min_lambda
        self.base = base
        self.gamma = gamma
        self.power = power

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.input_spec = base.InputSpec(min_ndim=2)
    
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Margin_Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        dtype=self.dtype, 
                                        trainable=True)

        self.built = True

    def call(self, inputs, labels, global_step):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        # norms
        inputs_norm = tf.norm(inputs, axis=1, keep_dims=True, name='inputs_norm')
        kernel_norm = tf.norm(self.kernel, axis=0, keep_dims=True, name='kernel_norm')
        # output inference
        outputs = tf.matmul(inputs, self.kernel/kernel_norm)
        theta = tf.acos(outputs/inputs_norm)
        # shape
        output_shape = outputs.get_shape().as_list()
        # cos_theta
        nd_indices = tf.stack([tf.range(output_shape[0], dtype=tf.int32), labels], axis=1)
        label_outputs = tf.gather_nd(outputs, nd_indices)
        label_theta = tf.gather_nd(theta, nd_indices)
        # beta
        beta = tf.maximum(self.min_lambda, self.base*tf.pow(1+self.gamma*tf.cast(global_step, tf.float32), self.power))
        # sign
        k = tf.floor(self.m*label_theta/math.pi)
        x_phi_theta = (tf.squeeze(inputs_norm)*(tf.cos(self.m*label_theta+k*math.pi)-2.*k) + beta*label_outputs)/(beta+1.0)
        # m_theta update
        m_outputs = tf.scatter_nd(nd_indices, x_phi_theta-label_outputs, shape=output_shape)
        outputs += m_outputs

        if self.activation is not None:
            return self.activation(outputs)
        return outputs, beta

    def _compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

def margin_fully_connected(inputs, labels, 
                           global_step,
                           num_outputs, 
                           m=4, 
                           min_lambda=5.0, 
                           base=1000, 
                           gamma=0.12, 
                           power=-1,
                           activation=None,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           weights_regularizer=None,
                           trainable=True,
                           name=None,
                           reuse=None):
    layer = Margin_Dense(num_outputs, m, 
                         min_lambda, base, gamma, power,
                         activation=activation,
                         kernel_initializer=weights_initializer,
                         kernel_regularizer=weights_regularizer,
                         trainable=trainable,
                         name=name,
                         dtype=inputs.dtype.base_dtype,
                         _scope=name,
                         _reuse=reuse)

    return layer.apply(inputs, labels, global_step)
