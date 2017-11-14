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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import math
import time
import numpy as np
from scipy.io import savemat

import tensorflow as tf
from data import eval_inputs
from nets.factory import net_select

parser = argparse.ArgumentParser()
# Visualization
parser.add_argument('--visual_embedding', type=int, default=0, 
                    help='Flags to generate embedding visualization.')
# Name configures
parser.add_argument('--net_name', type=str,
                    help='Name of the network architecture.')
parser.add_argument('--model_name', type=str,
                    help='Name of the training model.')
parser.add_argument('--fea_name', type=str,
                    help='Prefix name of feature files.')
# Directory configures
parser.add_argument('--eval_dir', type=str, default='eval',
                    help='Root directory where to write event logs.')
parser.add_argument('--feature_dir', type=str, default='features',
                    help='Root directory where to save feature files.')
parser.add_argument('--model_dir', type=str, default='models',
                    help='Root directory where checkpoints saved.')
# Data configures
parser.add_argument('--input_height', type=int, default=128,
                    help='The height of input images.')
parser.add_argument('--input_width', type=int, default=128,
                    help='The width of input images.')
parser.add_argument('--is_color', type=int, default=1, 
                    help='Whether to read inputs as RGB images.')
parser.add_argument('--flip_flag', type=int, default=0, 
                    help='Flip for flipped output.')
parser.add_argument('--data_list_path', type=str,
                    help='Path to the list of testing data.')
# Hyperparameters configures
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of images to process in a batch.')
FLAGS = parser.parse_args()


def evaluate():
  """Eval deep network for a number of steps."""
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    images, num_images = eval_inputs(FLAGS.data_list_path, 
                                     batch_size=FLAGS.batch_size, 
                                     input_height=FLAGS.input_height, 
                                     input_width=FLAGS.input_width, 
                                     is_color=FLAGS.is_color)
    with tf.device('/gpu:0'):
      network = net_select(FLAGS.net_name)
      features = network(images)
      if FLAGS.flip_flag:
        features_flipped = network(tf.reverse(images, axis=[2]), reuse=True)
        features = tf.concat([features, features_flipped], axis=1)
    # Session start
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    # Restore checkpoint
    ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.model_dir, FLAGS.net_name+'_'+FLAGS.model_name))
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      step = ckpt.model_checkpoint_path.split('-')[-1]
    else:
      raise IOError('No checkpoint file found')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    print('Extracting features from model saved in iteration %s...' % step)
    # Feature extraction
    start_time = time.time()
    wfea = sess.run(features)
    print('%d/%d features extracted... %.2fms elapsed'%(wfea.shape[0] if wfea.shape[0]<=num_images else num_images,
                                                       num_images, (time.time() - start_time)*1000))

    while wfea.shape[0] < num_images:
      start_time = time.time()
      fea = sess.run(features)
      wfea = np.vstack((wfea, fea))
      print('%d/%d features extracted... %.2fms elapsed'%(wfea.shape[0] if wfea.shape[0]<=num_images else num_images, 
                                                         num_images, (time.time() - start_time)*1000))

    # Only save num_images features
    wfea = wfea[0:num_images, :]
    print('Totally extracted %d features.'%(wfea.shape[0]))

    # Save features as .mat format files    
    print('Saving features to .mat files...')
    savemat(os.path.join(FLAGS.feature_dir, FLAGS.net_name+'_'+FLAGS.model_name, FLAGS.fea_name+'_'+step+'.mat'), {'wfea': wfea})

    # Embedding Visualization
    if FLAGS.visual_embedding:
      from PIL import Image
      from tensorflow.contrib.tensorboard.plugins import projector
      # Configures
      print('Creating visualized embedding...')
      embedding_var = tf.Variable(wfea, name='Embedding_128')
      sess.run(embedding_var.initializer)
      project_writer = tf.summary.FileWriter(os.path.join(FLAGS.eval_dir, FLAGS.net_name+'_'+FLAGS.model_name))
      config = projector.ProjectorConfig()
      embedding = config.embeddings.add()
      embedding.tensor_name = embedding_var.name

      metadata_path = os.path.join(FLAGS.eval_dir, FLAGS.net_name+'_'+FLAGS.model_name, FLAGS.fea_name+'_metadata.tsv')
      sprite_path = os.path.join(FLAGS.eval_dir, FLAGS.net_name+'_'+FLAGS.model_name, FLAGS.fea_name+'_sprite.png')
      if not (os.path.isfile(metadata_path) and os.path.isfile(sprite_path)):
        image_list = []
        label_list = []
        for path in open(os.path.expanduser(FLAGS.data_list_path), 'r'):
          image_path = path.strip('\n').split(' ')[0]
          image_list.append(image_path)
          label_list.append(image_path.split('/')[-2])

        # Create metadata
        with open(metadata_path, 'w') as meta:
          meta.write('Name\tClass\n')
          for idx, label in enumerate(label_list):
            meta.write('%06d\t%s\n' % (idx, label))

        # Create sprite
        single_dim = 32
        num_images_size = 50
        sprite_img = Image.new(mode='RGB', 
                               size=(num_images_size*single_dim, num_images_size*single_dim), 
                               color=(0, 0, 0))
        for idx, path in enumerate(image_list):
          if idx == num_images_size*num_images_size:
            break
          img = Image.open(path)
          img.thumbnail((single_dim, single_dim))
          idx_row = int(idx / num_images_size)
          idx_col = idx % num_images_size
          sprite_img.paste(img, (idx_col*single_dim, idx_row*single_dim))
          print('%d/%d image added to the sprite.' % (idx, num_images))
        sprite_img.save(sprite_path)
      
      embedding.metadata_path = FLAGS.fea_name+'_metadata.tsv'
      embedding.sprite.image_path = FLAGS.fea_name+'_sprite.png'
      embedding.sprite.single_image_dim.extend([64, 64])

      projector.visualize_embeddings(project_writer, config)
      saver = tf.train.Saver([embedding_var])
      saver.save(sess, os.path.join(FLAGS.eval_dir, FLAGS.net_name+'_'+FLAGS.model_name, 'embedding.ckpt'), 1)

    print('Done.')

    

def main(argv=None):
  tf.gfile.MakeDirs(os.path.join(FLAGS.feature_dir, FLAGS.net_name+'_'+FLAGS.model_name))
  evaluate()

if __name__ == '__main__':
  tf.app.run()
