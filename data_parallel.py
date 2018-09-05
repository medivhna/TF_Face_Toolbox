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
from tensorflow.contrib import nccl
from tensorflow.contrib.layers import fully_connected, l2_regularizer
from nets.model_parallel import *

from collections import OrderedDict


class Singular(object):
  def __init__(self, model, lr, optimizer, weight_decay=5e-4):
    self.model = model
    self.lr = lr
    self.optimizer = optimizer
    self.weight_decay = weight_decay
    self.num_gpus = 1

  def _grad_var(self, total_loss, params, mult_lr=1.0):
    grads = tf.gradients(total_loss, params, aggregation_method=tf.AggregationMethod.DEFAULT)
    for grad, var in list(zip(grads, params)):
      if grad is None:
        print(grad, var)
    grads = [mult_lr*grad*(1.0/self.num_gpus) for grad in grads]
    grads_vars = list(zip(grads, params))
    for grad, var in grads_vars:
      if grad is not None:
        tf.summary.histogram(var.name, var)
        tf.summary.histogram(var.op.name + '/gradients', grad)
    return grads_vars

  def __call__(self, inputs):
    # Inference 
    with tf.name_scope('TOWER') as name_scope:
      with tf.device('/gpu:0'):
        # Forward
        #logits = self.model.forward(inputs['images'], num_classes=inputs['num_cameras'], is_training=True)
        logits = self.model.forward(inputs['images'], num_classes=inputs['num_classes'], is_training=True)
        self.pretrained_param = self.model.pretrained_param()
        # Losses
        #losses, losses_name, others = self.model.loss_function(name_scope, inputs['cam_ids'], **logits)
        losses, losses_name, others = self.model.loss_function(name_scope, inputs['labels'], **logits)
        total_loss = tf.add_n(losses, name='total_loss')
        print('Model has been inferenced.')
        # Params & Gradients
        mult_lr_list = self.model.mult_lr_list()
        params = self.model.param_list(is_training=True, trainable=True)
        total_grads_vars = []
        for mult_lr, param in zip(mult_lr_list, params):
          total_grads_vars += self._grad_var(total_loss, param, mult_lr)
        # Optimizer settings
        if self.optimizer == 'Momentum':
          opt = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
        elif self.optimizer == 'Adam':
          opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)    
        train_op = opt.apply_gradients(total_grads_vars)
        print('Optimizer has been configured.')

        # BN updates
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
        
    global_step = tf.train.get_global_step()
    global_step_op = global_step.assign_add(1)
    train_ops = tf.group(*([train_op, update_ops, global_step_op]))

    return train_ops, losses, losses_name, others

class DataParallel(Singular):
  def __init__(self, model, lr, optimizer, num_gpus=4, weight_decay=5e-4):
    assert num_gpus > 1, 'DataParallel objects are only used for multi-gpu training tasks.'
    super(DataParallel, self).__init__(model, lr, optimizer, weight_decay)
    self.num_gpus = num_gpus
    self.pretrained_param = []

  def _reduced_opt(self, tower_grads_vars):
    tower_reduced_grads_vars = []
    for grads_vars in zip(*tower_grads_vars):
      grads = [g for g, _ in grads_vars]
      reduced_grads = nccl.all_sum(grads)
      reduced_grads_vars = [(g, v) for (_, v), g in zip(grads_vars, reduced_grads)]
      tower_reduced_grads_vars.append(reduced_grads_vars)

    # Optimizier
    tower_train_ops = []
    grad_state = [list(x) for x in zip(*tower_reduced_grads_vars)]
    for device_id in xrange(self.num_gpus):
      with tf.device('/gpu:%d' % device_id):
        # Gradients of TOWER_(device_id)
        grads = grad_state[device_id]
        # Optimizer configure
        if self.optimizer == 'Momentum':
          opt = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
        elif self.optimizer == 'Adam':
          opt = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999)
        # Tower train_ops
        tower_train_ops.append(opt.apply_gradients(grads))

        print('Optimizer %d has been configured.' % device_id)

    return tower_train_ops, tower_reduced_grads_vars


  def __call__(self, inputs):
    # Inputs
    # Split the input batch into num_gpus sub-batch (Data parallel)
    images_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['images'])
    labels_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['labels'])
    #cam_id_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['cam_ids'])

    # Inference 
    tower_grads_vars = []
    tower_losses = []
    tower_others = OrderedDict()

    for device_id in xrange(self.num_gpus):
      with tf.variable_scope('replicated_%s' % device_id) as scope:
        with tf.name_scope('TOWER_%d' % device_id) as name_scope:
          with tf.device('/gpu:%d' % device_id):
            # Forward
            logits = self.model.forward(images_splits[device_id], num_classes=inputs['num_classes'], is_training=True)
            #logits = self.model.forward(images_splits[device_id], num_classes=inputs['num_classes'], num_cameras=inputs['num_cameras'], is_training=True)
            self.pretrained_param += self.model.pretrained_param(scope=scope)
            # Losses
            losses, losses_name, others = self.model.loss_function(name_scope, labels_splits[device_id], **logits)
            #losses, losses_name, others = self.model.loss_function(name_scope, labels_splits[device_id], cam_id_splits[device_id], **logits)
            total_loss = tf.add_n(losses, name='total_loss')

            # Variables & Gradients
            mult_lr_list = self.model.mult_lr_list(scope)
            params = self.model.param_list(is_training=True, trainable=True, scope=scope)
            grads_vars_inner = []
            for mult_lr, param in zip(mult_lr_list, params):
              grads_vars_inner += self._grad_var(total_loss, param, mult_lr)

            # Tower grads, losses
            tower_grads_vars.append(grads_vars_inner)
            tower_losses.append(losses)
            # BN updates
            if device_id == 0:
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            print('Tower %d has been inferenced.' % device_id)

    # Allreduce losses
    allreduce_losses = [tf.add_n(losses)/self.num_gpus for losses in zip(*tower_losses)]
    # Allreduce gradients
    tower_train_ops, tower_reduced_grads_vars = self._reduced_opt(tower_grads_vars)

    global_step = tf.train.get_global_step()
    global_step_op = global_step.assign_add(1)
    train_ops = tf.group(*(tower_train_ops+update_ops+[global_step_op]))

    return train_ops, allreduce_losses, losses_name, others

class DataParallel_margin(Singular):
  def __init__(self, model, lr, optimizer, num_gpus=4, weight_decay=5e-4):
    assert num_gpus > 1, 'DataParallel objects are only used for multi-gpu training tasks.'
    super(DataParallel_margin, self).__init__(model, lr, optimizer, weight_decay)
    self.num_gpus = num_gpus
    self.pretrained_param = []

  def _reduced_opt(self, tower_grads_vars):
    tower_reduced_grads_vars = []
    for grads_vars in zip(*tower_grads_vars):
      grads = [g for g, _ in grads_vars]
      reduced_grads = nccl.all_sum(grads)
      reduced_grads_vars = [(g, v) for (_, v), g in zip(grads_vars, reduced_grads)]
      tower_reduced_grads_vars.append(reduced_grads_vars)

    # Optimizier
    tower_train_ops = []
    grad_state = [list(x) for x in zip(*tower_reduced_grads_vars)]
    for device_id in xrange(self.num_gpus):
      with tf.device('/gpu:%d' % device_id):
        # Gradients of TOWER_(device_id)
        grads = grad_state[device_id]
        # Optimizer configure
        if self.optimizer == 'Momentum':
          opt = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
        elif self.optimizer == 'Adam':
          opt = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999)
        # Tower train_ops
        tower_train_ops.append(opt.apply_gradients(grads))

        print('Optimizer %d has been configured.' % device_id)

    return tower_train_ops, tower_reduced_grads_vars


  def __call__(self, inputs):
    # Inputs
    # Split the input batch into num_gpus sub-batch (Data parallel)
    images_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['images'])
    labels_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['labels'])
    #cam_id_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=inputs['cam_ids'])

    # Inference 
    tower_grads_vars = []
    tower_losses = []
    tower_others = {}

    for device_id in xrange(self.num_gpus):
      with tf.variable_scope('replicated_%s' % device_id) as scope:
        with tf.name_scope('TOWER_%d' % device_id) as name_scope:
          with tf.device('/gpu:%d' % device_id):
            # Forward
            logits = self.model.forward(images_splits[device_id], labels_splits[device_id], num_classes=inputs['num_classes'], is_training=True)
            self.pretrained_param += self.model.pretrained_param(scope=scope)
            # Losses
            losses, losses_name, others = self.model.loss_function(name_scope, labels_splits[device_id], **logits)
            total_loss = tf.add_n(losses, name='total_loss')
            for key, val in others.items():
              if not tower_others.has_key(key):
                tower_others[key] = []
              tower_others[key].append(val)


            # Variables & Gradients
            mult_lr_list = self.model.mult_lr_list(scope)
            params = self.model.param_list(is_training=True, trainable=True, scope=scope)
            grads_vars_inner = []
            for mult_lr, param in zip(mult_lr_list, params):
              grads_vars_inner += self._grad_var(total_loss, param, mult_lr)

            # Tower grads, losses
            tower_grads_vars.append(grads_vars_inner)
            tower_losses.append(losses)
            # BN updates
            if device_id == 0:
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            print('Tower %d has been inferenced.' % device_id)

    # Allreduce losses
    allreduce_losses = [tf.add_n(losses)/self.num_gpus for losses in zip(*tower_losses)]
    # Allreduce gradients
    tower_train_ops, tower_reduced_grads_vars = self._reduced_opt(tower_grads_vars)

    global_step = tf.train.get_global_step()
    global_step_op = global_step.assign_add(1)
    train_ops = tf.group(*(tower_train_ops+update_ops+[global_step_op]))

    return train_ops, allreduce_losses, losses_name, tower_others