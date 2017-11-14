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

import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib import nccl
from tensorflow.contrib.layers import fully_connected, l2_regularizer

def loss_function(logits, labels, scope=None):
  losses = []
  losses_name = []

  # cross_entropy_loss
  cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
  cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
  losses.append(cross_entropy_loss)
  losses_name.append('cross_entropy')

  # TODO: Other losses

  # regularization_loss
  regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope), name='reg_loss')
  losses.append(regularization_loss)
  losses_name.append('regularization')

  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses)
  for l in losses:
    loss_name = re.sub('TOWER_[0-9]*/', '', l.op.name)
    tf.summary.scalar(loss_name +' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))

  return losses, losses_name

class DataParallel(object):
  # Split the input batch into num_gpus sub-batch (Data parallel)

  def __init__(self, model, init_lr, decay_epoch, decay_rate, batches_per_epoch, num_gpus=1):
    self.model = model
    self.num_gpus = num_gpus

    # Learning rate configure
    decay_boundary = [(np.int64(epoch)-1)*batches_per_epoch for epoch in decay_epoch.split(',')]
    decay_value = [init_lr]+[init_lr*decay_rate**(p+1) for p in xrange(len(decay_boundary))]
    self.lr = tf.train.piecewise_constant(tf.train.get_global_step(), decay_boundary, decay_value)
    tf.summary.scalar('learning_rate', self.lr)

  def __call__(self, **inputs):
    # Inputs
    images_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['images'])
    labels_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['labels'])

    # Inference 
    tower_grads = []
    tower_losses = []
    for device_id in xrange(self.num_gpus):
      with tf.variable_scope('replicated_%s' % device_id):
        with tf.name_scope('TOWER_%d' % device_id) as name_scope:
          with tf.device('/gpu:%d' % device_id):
            # Forward
            pre_logits = self.model(images_splits[device_id], is_training=True)
            logits = fully_connected(pre_logits, num_outputs=inputs['num_classes'], 
                                     activation_fn=None, biases_initializer=None,
                                     weights_regularizer=l2_regularizer(0.0005))
            # Losses
            losses, losses_name = loss_function(logits, labels_splits[device_id], scope=name_scope)
            total_loss = tf.add_n(losses, name='total_loss')

            # Variables 
            params = [v for v in tf.trainable_variables() if v.name.startswith('replicated_%s/' % device_id)]

            # Gradients
            grads = tf.gradients(total_loss, params, aggregation_method=tf.AggregationMethod.DEFAULT)
            grads = [grad/self.num_gpus for grad in grads]

            gradvars = list(zip(grads, params))

            for grad, var in gradvars:
              if grad is not None:
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(var.op.name + '/gradients', grad)

            # Tower grads, losses and updates
            tower_grads.append(gradvars)
            tower_losses.append(losses)
            if device_id == 0:
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            print('Tower %d has been inferenced.' % device_id)

    # Allreduce losses
    allreduce_losses = [tf.add_n(losses)/self.num_gpus for losses in zip(*tower_losses)]

    # Allreduce gradients
    allreduce_grads = []
    for grad_and_vars in zip(*tower_grads):
      grads = [g for g, _ in grad_and_vars]
      summed_grads = nccl.all_sum(grads)
      new_grads_and_vars = [(g, v) for (_, v), g in zip(grad_and_vars, summed_grads)]
      allreduce_grads.append(new_grads_and_vars)
    grad_state = [list(x) for x in zip(*allreduce_grads)]

    # Optimizier
    tower_train_ops = []
    for device_id in xrange(self.num_gpus):
      with tf.device('/gpu:%d' % device_id):
        # Gradients of TOWER_(device_id)
        grads = grad_state[device_id]
        # Optimizer configure
        opt = tf.train.MomentumOptimizer(self.lr, 0.9)
        # Tower train_ops
        tower_train_ops.append(opt.apply_gradients(grads))

        print('Optimizer %d has been configured.' % device_id)

    global_step = tf.train.get_global_step()
    global_step_op = global_step.assign_add(1)
    train_ops = tf.group(*(tower_train_ops+update_ops+[global_step_op]))

    return train_ops, self.lr, allreduce_losses, losses_name

  def vars(self):
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'replicated_0' or not v.name.startswith('replicated_'):
        params.append(v)

    return params
