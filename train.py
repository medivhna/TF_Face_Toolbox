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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import time
from datetime import datetime

import tensorflow as tf
from utils.gpu_select import gpu_select

from data import train_inputs
from saver import DataParallelSaverBuilder
from nets.net_base import net_select


parser = argparse.ArgumentParser()
# Name configures
parser.add_argument('--net_name', type=str,
                    help='Name of the network architecture.')
parser.add_argument('--model_name', type=str,
                    help='Name of the training model.')
# Directory configures
parser.add_argument('--train_dir', type=str, default='train',
                    help='Root directory where to write event logs.')
parser.add_argument('--model_dir', type=str, default='models',
                    help='Root directory where to save checkpoints.')
parser.add_argument('--pretrained_path', type=str, default='',
                    help='Path to save pretrained checkpoints.')
# Data configure
parser.add_argument('--data_format', type=str, default='NCHW',
                    help='The format of data in the network (NCHW(default) or NHWC).')
parser.add_argument('--train_list_path', type=str, 
                    help='Path to the list of training data.')
parser.add_argument('--input_height', type=int, default=384,
                    help='The height of input images.')
parser.add_argument('--input_width', type=int, default=128,
                    help='The width of input images.')
parser.add_argument('--crop_height', type=int, default=-1,
                    help='The height of input images.')
parser.add_argument('--crop_width', type=int, default=-1,
                    help='The width of input images.')
parser.add_argument('--is_color', type=int, default=1, 
                    help='Whether to read inputs as RGB images.')
parser.add_argument('--augmentation', type=int, default=0, 
                    help='Whether to employ data augmentation to training set.')
# Hyperparameters configure
parser.add_argument('--batch_size', type=int, default=-1,
                    help='Number of sampled images in a batch.')
parser.add_argument('--num_classes', type=int, default=-1,
                    help='Number of sampled classesin a batch.')
parser.add_argument('--num_per_class', type=int, default=-1,
                    help='Number of sampled images per class in a batch.')
parser.add_argument('--optimizer', type=str, default='Momentum',
                    help='Type of optimizer.')
parser.add_argument('--init_lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--lr_decay_method', type=str, default='step',
                    help='Learning rate strategy (step/cosine/exp).')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                     help='Learning rate decay rate.')
parser.add_argument('--lr_decay_epoch', type=str, default='',
                     help='Boundaries of decaying learning rate in step lr_decay')
parser.add_argument('--max_epoches', type=int, 
                    help='Number of batches to run.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Factor for weight decaying.')
# Device configures
parser.add_argument('--num_gpus', type=int, default=4,
                    help='Number of GPUs to use.')
# Interval configures
parser.add_argument('--display_interval', type=int, default=10, 
                    help='Internal iterations of verbose.')
parser.add_argument('--save_interval', type=int, default=1000, 
                    help='Internal iterations of saving models.')
FLAGS = parser.parse_args()

 #and   and )
def FLAGS_assertion():
  assert FLAGS.data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
  assert FLAGS.batch_size != -1 or (FLAGS.num_classes != -1 and FLAGS.num_per_class != -1)
  assert (FLAGS.num_classes != -1 and FLAGS.num_per_class != -1 and (FLAGS.num_classes*FLAGS.num_per_class)%FLAGS.num_gpus==0) or (FLAGS.batch_size%FLAGS.num_gpus==0 and FLAGS.batch_size != -1)
  assert FLAGS.optimizer in ['Momentum', 'Adam'], 'Unsupported optimizer.'

def get_local_init_ops():
  local_var_init_op = tf.local_variables_initializer()
  local_init_ops = [local_var_init_op]

  global_vars = tf.global_variables()
  var_by_name = dict([(v.name, v) for v in global_vars])
  copy_ops = []
  for v in global_vars:
    split_name = v.name.split('/')
    if split_name[0] == 'replicated_0' or split_name[-1].startswith('avg') or not v.name.startswith('replicated_'):
      continue
    split_name[0] = 'replicated_0'
    copy_from = var_by_name['/'.join(split_name)]
    copy_ops.append(v.assign(copy_from.read_value()))

  with tf.control_dependencies([local_var_init_op]):
    local_init_ops.extend(copy_ops)
  local_init_op_group = tf.group(*local_init_ops)

  return local_init_op_group

def lr_config(method, batches_per_epoch):
  if method == 'step':
    import numpy as np
    if FLAGS.lr_decay_epoch == '':
      raise ValueError('Empty learning rate decay epoch boundaries.')
    decay_boundary = [(np.int64(epoch)-1)*batches_per_epoch for epoch in FLAGS.lr_decay_epoch.split(',')]
    decay_value = [FLAGS.init_lr]+[FLAGS.init_lr*FLAGS.lr_decay_rate**(p+1) for p in xrange(len(decay_boundary))]
    lr = tf.train.piecewise_constant(tf.train.get_global_step(), decay_boundary, decay_value)
  elif method == 'exp':
    global_step = tf.train.get_global_step()
    decay_step = int(FLAGS.lr_decay_epoch)*batches_per_epoch
    lr = tf.cond(tf.less(global_step, decay_step),
                 lambda: FLAGS.init_lr,
                 lambda: tf.train.exponential_decay(FLAGS.init_lr, 
                                                    global_step-decay_step, 
                                                    decay_steps=int(FLAGS.max_epoches)*batches_per_epoch+1-decay_step,
                                                    decay_rate=0.001))
  elif method == 'cosine':
    lr = tf.train.cosine_decay(FLAGS.init_lr, tf.train.get_global_step(), FLAGS.max_epoches*batches_per_epoch)
  else:
    raise ValueError('Unsupported learning rate decaying method.')

  return lr

def format_str(losses_name):
  outputs = '[%s] Epoch/Step %d/%d, lr = %g\n'
  for loss_id, loss_name in enumerate(losses_name):
    outputs += '[%s]    Loss #'+str(loss_id)+': '+loss_name+' = %.6f\n'
  outputs += '[%s]    batch_time = %.1fms/batch, throughput = %.1fimages/s'

  return outputs

def train():
  # Global graph
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()

    # Model builder
    ## Data I/O
    inputs = train_inputs(FLAGS.train_list_path, 
                          input_height=FLAGS.input_height, 
                          input_width=FLAGS.input_width,
                          crop_height=FLAGS.crop_height,
                          crop_width=FLAGS.crop_width,
                          is_color=FLAGS.is_color,
                          augmentation=FLAGS.augmentation,
                          batch_size=FLAGS.batch_size,
                          num_classes=FLAGS.num_classes,
                          num_per_class=FLAGS.num_per_class)
    batch_size = FLAGS.batch_size if FLAGS.batch_size != -1 else FLAGS.num_classes*FLAGS.num_per_class
    batches_per_epoch = inputs['num_examples'] // batch_size + 1
    ## Network selection
    network = net_select(FLAGS.net_name, FLAGS.data_format, FLAGS.weight_decay)
    ## Learning rate config
    lr = lr_config(FLAGS.lr_decay_method, batches_per_epoch)
    ## Parallel method according to num_gpus
    if FLAGS.num_gpus > 1:
      from data_parallel import DataParallel_margin
      model = DataParallel_margin(network, lr, optimizer=FLAGS.optimizer, weight_decay=FLAGS.weight_decay, num_gpus=FLAGS.num_gpus)
    else:
      from data_parallel import Singular
      model = Singular(network, lr, optimizer=FLAGS.optimizer, weight_decay=FLAGS.weight_decay)
    train_ops, losses, losses_name, others = model(inputs)

    # Saver configs
    ## Checkpoint saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20, 
                          builder=DataParallelSaverBuilder() if FLAGS.num_gpus > 1 else None)
    ## Finetune Saver
    if FLAGS.pretrained_path != '':
      ft_saver = tf.train.Saver(model.pretrained_param, 
                                builder=DataParallelSaverBuilder() if FLAGS.num_gpus > 1 else None)

    # Supervisor
    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.train_dir, FLAGS.net_name+'_'+FLAGS.model_name),
                             local_init_op=get_local_init_ops(),
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=0)
    config=tf.ConfigProto(allow_soft_placement=True, 
                          log_device_placement=False, 
                          gpu_options=tf.GPUOptions(allow_growth=True))    

    # Training session
    with sv.managed_session(config=config) as sess:
      ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.model_dir, FLAGS.net_name+'_'+FLAGS.model_name))
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored from %s' % os.path.join(FLAGS.model_dir, FLAGS.net_name+'_'+FLAGS.model_name))
      elif FLAGS.pretrained_path != '':
        ft_saver.restore(sess, FLAGS.pretrained_path)
        print('Network parameters initialized from %s' % FLAGS.pretrained_path)
      else:
        print('Network parameters initialized from scratch.')

      print('%s training start...' % (FLAGS.net_name+'_'+FLAGS.model_name))
      step = 0
      epoch = 1

      time_sim = 0
      image_sim = 0
      while epoch <= FLAGS.max_epoches:
        step = sess.run(global_step)
        epoch = step // batches_per_epoch + 1
        
        start_time = time.time()
        _, learning_rate, losses_value, others_dict = sess.run([train_ops, lr, losses, others])
        duration = time.time() - start_time

        if step % FLAGS.display_interval == 0:
          # Format tuple
          format_list = [datetime.now(), epoch, step, learning_rate]
          for loss_value in losses_value:
            format_list.extend([datetime.now(), loss_value])
          format_list.extend([datetime.now(), duration * 1000, batch_size / duration])
          print(format_str(losses_name) % tuple(format_list))
          for other_name, other_value in others_dict.items():
            print('%s: %s' % (other_name, other_value) )

        if step > 0:
          time_sim += duration
          image_sim += batch_size/duration

        if (step > 0 and step % FLAGS.save_interval == 0) or step == FLAGS.max_epoches*batches_per_epoch:
          train_path = os.path.join(FLAGS.model_dir, FLAGS.net_name+'_'+FLAGS.model_name, FLAGS.net_name+'_'+FLAGS.model_name+'.ckpt')
          saver.save(sess, train_path, global_step=step)
          print('[%s]: Model has been saved in Iteration %d' % (datetime.now(), step))

      print('mean batch_time=%.2f, mean throughput=%.2f'%(time_sim/step*1000, image_sim/step))

def main(argv=None):
  FLAGS_assertion()
  tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, FLAGS.net_name+'_'+FLAGS.model_name))
  tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, FLAGS.net_name+'_'+FLAGS.model_name))
  train()

if __name__ == '__main__':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu_select(FLAGS.num_gpus, wait_hour=0.5, wait_for_long=True)
  tf.app.run()
