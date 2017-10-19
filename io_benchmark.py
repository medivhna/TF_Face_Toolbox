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

import os.path
import time
import importlib
from datetime import datetime

import tensorflow as tf
from nets.resnext import ResNeXt
from utils.data import train_inputs
from utils.parallel import DataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

FLAGS = tf.app.flags.FLAGS

# Device configures
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('num_threads_per_gpu', 8,
                            """How many threads to use for per GPU to prefetch images.""")

tf.app.flags.DEFINE_integer('display_interval', 10, 
                            """Internal iterations of verbose.""")
tf.app.flags.DEFINE_integer('save_interval', 1000, 
                            """Internal iterations of saving models.""")

# Directory configures
tf.app.flags.DEFINE_string('model_name', 'ResNeXt_50_webface',
                           """Directory where to save network model.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('model_dir', 'models',
                           """Directory where to save checkpoint.""")
# Data configure
tf.app.flags.DEFINE_string('data_list_path', '/home/wangguanshuo/lists/WebFace/mtcnn_2.npy',
                           """Path to the data directory for training.""")
tf.app.flags.DEFINE_integer('input_size', 128,
                           """The size of input images.""")
tf.app.flags.DEFINE_boolean('is_color', True, 
                           """Input 3-channels images.""")

# Hyperparameters configure
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 28000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('init_lr', 0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_float('lr_decay_rate', 0.1,
                            """Decay rate of the learning rate.""")
tf.app.flags.DEFINE_string('lr_decay_step', '120000,240000,340000',
                            """Decay rate of the learning rate.""")

def train():
    with tf.Graph().as_default():
        # Global graph and data 
        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()
            global_step = tf.cast(global_step, dtype=tf.int32)

            inputs = train_inputs(FLAGS.data_list_path, FLAGS.batch_size, FLAGS.is_color, 
                                  input_size=FLAGS.input_size, augment=False,
                                  num_preprocess_threads=FLAGS.num_threads_per_gpu*FLAGS.num_gpus)

        # Saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.model_dir, FLAGS.model_name))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from %s' % os.path.join(FLAGS.model_dir, FLAGS.model_name))
        else:
            print('Network parameters initialized from scratch.')

        # Supervisor
        sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.train_dir, FLAGS.model_name),
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=0)
        # Session config
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                gpu_options=tf.GPUOptions(allow_growth=True))

        # Training session
        with sv.managed_session(config=config) as sess:
            tf.train.start_queue_runners(sess=sess)

            while True:
                start_time = time.time()
                images, labels = sess.run([inputs['images'], inputs['labels']])
                duration = time.time() - start_time
                print('I/O time: %.2fms\n' % duration*1000)

def main(argv=None):
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, FLAGS.model_name))
    tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, FLAGS.model_name))
    train()

if __name__ == '__main__':
    tf.app.run()
