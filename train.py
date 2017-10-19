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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import time
from datetime import datetime

import tensorflow as tf
from nets.resnext import ResNeXt
from utils.data import train_inputs
from utils.saver import DataParallelSaverBuilder
from utils.parallel import DataParallel

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
parser = argparse.ArgumentParser()
# Directory configures
parser.add_argument('--train_dir', type=str, default='train',
                    help='Root directory where to write event logs.')
parser.add_argument('--model_dir', type=str, default='models',
                    help='Root directory where to save checkpoints.')
parser.add_argument('--model_name', type=str,
                    help='Name of the training model.')
# Data configure
parser.add_argument('--data_format', type=str, default='NCHW',
                    help='The format of data in the network (NCHW(default) or NHWC).')
parser.add_argument('--input_size', type=int, default=128,
                    help='The size of input images.')
parser.add_argument('--is_color', type=bool, default=True, 
                    help='Whether to read inputs as RGB images.')
parser.add_argument('--data_list_path', type=str, 
                    help='Path to the list of training data.')
# Hyperparameters configure
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of images to process in a batch.')
parser.add_argument('--init_lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                    help='Learning rate decay rate.')
parser.add_argument('--lr_decay_epoch', type=str,
                    help='Boundaries of decaying learning rate.')
parser.add_argument('--max_epoches', type=int, 
                    help='Number of batches to run.')
# Device configures
parser.add_argument('--num_gpus', type=int, default=4,
                    help='Number of GPUs to use.')
parser.add_argument('--num_threads_per_gpu', type=int, default=8,
                    help='Number of threads to use for per GPU to prefetch images.')
# Interval configures
parser.add_argument('--display_interval', type=int, default=10, 
                    help='Internal iterations of verbose.')
parser.add_argument('--save_interval', type=int, default=1000, 
                    help='Internal iterations of saving models.')
FLAGS = parser.parse_args()

def FLAGS_assertion():
    assert FLAGS.data_format in ['NCHW', 'NHWC'], 'Unknown data format.'
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')

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

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Global graph
        global_step = tf.train.get_or_create_global_step()
        # Data
        images, labels, num_classes, num_examples = train_inputs(FLAGS.data_list_path, 
                                                                 FLAGS.batch_size, 
                                                                 FLAGS.is_color, 
                                                                 input_size=FLAGS.input_size, augment=False,
                                                                 num_preprocess_threads=FLAGS.num_threads_per_gpu*FLAGS.num_gpus)
        batches_per_epoch = num_examples // FLAGS.batch_size + 1

        # Network
        network = ResNeXt(num_layers=50, num_card=1, data_format=FLAGS.data_format)

        # DataParallel
        model = DataParallel(network, 
                             init_lr=FLAGS.init_lr, 
                             decay_epoch=FLAGS.lr_decay_epoch, 
                             decay_rate=FLAGS.lr_decay_rate,
                             batches_per_epoch=batches_per_epoch, 
                             num_gpus=FLAGS.num_gpus)

        # Inference
        train_ops, lr, losses, losses_name = model(images=images, labels=labels, num_classes=num_classes)

        # Saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20,
                               builder=DataParallelSaverBuilder())

        # Supervisor
        sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.train_dir, FLAGS.model_name),
                                 local_init_op=get_local_init_ops(),
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=0)

        # Session config
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                gpu_options=tf.GPUOptions(allow_growth=True))
        run_metadata = tf.RunMetadata()

        # Format string 
        format_str = '[%s] Epoch/Step %d/%d, lr = %g\n'
        for loss_id, loss_name in enumerate(losses_name):
            format_str += '[%s]      Loss #'+str(loss_id)+': '+loss_name+' = %.6f\n'
        format_str += '[%s]      batch_time = %.1fms/batch, throughput = %.1fimages/s'

        # Training session
        with sv.managed_session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.model_dir, FLAGS.model_name))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored from %s' % os.path.join(FLAGS.model_dir, FLAGS.model_name))
            else:
                print('Network parameters initialized from scratch.')

            print('%s training start...' % FLAGS.model_name)
            # I/O ops
            tf.train.start_queue_runners(sess=sess)

            step = 0
            epoch = 1
            while epoch <= FLAGS.max_epoches:
                step = sess.run(global_step)
                epoch = step // batches_per_epoch + 1

                start_time = time.time()
                output = sess.run([train_ops, lr]+losses)
                learning_rate = output[1]
                losses_value = output[2:]
                duration = time.time() - start_time


                if step % FLAGS.display_interval == 0:
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = duration * 1000

                    # Format tuple
                    format_list = [datetime.now(), epoch, step, learning_rate]
                    for loss_value in losses_value:
                        format_list.extend([datetime.now(), loss_value])
                    format_list.extend([datetime.now(), sec_per_batch, examples_per_sec])
                    print(format_str % tuple(format_list))

                    if (step > 0 and step % FLAGS.save_interval == 0) or step == FLAGS.max_epoches*batches_per_epoch:
                        train_path = os.path.join(FLAGS.model_dir, FLAGS.model_name, FLAGS.model_name+'.ckpt')
                        saver.save(sess, train_path, global_step=step)
                        print('[%s]: Model has been saved in Iteration %d' % (datetime.now(), step))

def main(argv=None):
    FLAGS_assertion()
    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, FLAGS.model_name))
    tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, FLAGS.model_name))
    train()

if __name__ == '__main__':
    tf.app.run()


                # = sess.run(losses)
                # if step < 10:
                #   sess.run([train_ops, lr]+losses)
                # else:
                #     from tensorflow.python.client import timeline
                #     sess.run([train_ops, lr]+losses,
                #              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                #                                    run_metadata=run_metadata)
                #     trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                #     with open('timeline5.ctf.json', 'w') as trace_file:
                #         trace_file.write(trace.generate_chrome_trace_format())
                #     print('Done.')
                #     return 
                #print('step %d: %.2fms' % (step, duration*1000))