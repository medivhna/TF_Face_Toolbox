from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import math
import time
import numpy as np
from PIL import Image
from scipy.io import savemat

import tensorflow as tf
from data import eval_inputs
from nets.resnext import ResNeXt

parser = argparse.ArgumentParser()
# Visualization
parser.add_argument('--visual_embedding', type=bool, default=False, 
                    help='Flags to generate embedding visualization.')
# Directory configures
parser.add_argument('--eval_dir', type=str, default='eval',
                    help='Root directory where to write event logs.')
parser.add_argument('--feature_dir', type=str, default='features',
                    help='Root directory where to save feature files.')
parser.add_argument('--model_dir', type=str, default='models',
                    help='Root directory where checkpoints saved.')
parser.add_argument('--model_name', type=str,
                    help='Name of the pretrained model.')
# Data configures
parser.add_argument('--input_size', type=int, default=128,
                    help='The size of input images.')
parser.add_argument('--is_color', type=bool, default=True, 
                    help='Whether to read inputs as RGB images.')
parser.add_argument('--fea_name', type=str,
                    help='Prefix name of feature files.')
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
                                     input_size=FLAGS.input_size, 
                                     is_color=FLAGS.is_color)
    with tf.variable_scope('replicated_0'):
      with tf.device('/gpu:0'):
        network = ResNeXt(num_layers=50, num_card=1)
        features = network(images)
    # Session start
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    #sess.run(tf.global_variables_initializer())

    # Restore checkpoint
    ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.model_dir, FLAGS.model_name))
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
    savemat(os.path.join(FLAGS.feature_dir, FLAGS.model_name, FLAGS.fea_name+'_'+step+'.mat'), {'wfea': wfea})

    # Embedding Visualization
    if FLAGS.visual_embedding:
      from tensorflow.contrib.tensorboard.plugins import projector
      # Configures
      print('Creating visualized embedding...')
      embedding_var = tf.Variable(wfea, name='Embedding_128')
      sess.run(embedding_var.initializer)
      project_writer = tf.summary.FileWriter(os.path.join(FLAGS.eval_dir, FLAGS.model_name))
      config = projector.ProjectorConfig()
      embedding = config.embeddings.add()
      embedding.tensor_name = embedding_var.name

      metadata_path = os.path.join(FLAGS.eval_dir, FLAGS.model_name, FLAGS.fea_name+'_metadata.tsv')
      sprite_path = os.path.join(FLAGS.eval_dir, FLAGS.model_name, FLAGS.fea_name+'_sprite.png')
      if not (os.path.isfile(metadata_path) and os.path.isfile(sprite_path)):
        image_list = []
        label_list = []
        with open(FLAGS.data_list_path, 'r') as f:
          lines = f.readlines()
        for path in lines:
          image_list.append(path.strip('\n'))
          label_list.append(path.split('/')[-2])

        # Create metadata
        with open(metadata_path, 'w') as meta:
          meta.write('Name\tClass\n')
          for idx, label in enumerate(label_list):
            meta.write('%06d\t%s\n' % (idx, label))

        # Create sprite
        single_dim = 32
        num_images_size = int(math.sqrt(num_images))
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
      
      embedding.metadata_path = metadata_path
      embedding.sprite.image_path = sprite_path
      embedding.sprite.single_image_dim.extend([64, 64])

      projector.visualize_embeddings(project_writer, config)
      saver = tf.train.Saver([embedding_var])
      saver.save(sess, os.path.join(FLAGS.eval_dir, FLAGS.model_name, 'embedding.ckpt'), 1)

    print('Done.')

    

def main(argv=None):
  tf.gfile.MakeDirs(os.path.join(FLAGS.feature_dir, FLAGS.model_name))
  evaluate()

if __name__ == '__main__':
  tf.app.run()
