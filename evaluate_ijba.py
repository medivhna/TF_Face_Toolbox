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
from nets.factory import net_select

parser = argparse.ArgumentParser()
# Visualization
parser.add_argument('--visual_embedding', type=bool, default=False, 
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
parser.add_argument('--is_color', type=bool, default=True, 
                    help='Whether to read inputs as RGB images.')
parser.add_argument('--flip_flag', type=str, default=False, 
                    help='Flip for flipped output.')
parser.add_argument('--pattern', type=str, default='verify_metadata_',
                    help='Path to the list of testing data.')
parser.add_argument('--data_list_dir', type=str,
                    help='Path to the list of testing data.')
# Hyperparameters configures
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of images to process in a batch.')
FLAGS = parser.parse_args()


def evaluate(split):
  """Eval deep network for a number of steps."""
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    data_list_path = os.path.join(FLAGS.data_list_dir, FLAGS.pattern+str(split)+'.txt')
    images, num_images = eval_inputs(data_list_path, 
                                     batch_size=FLAGS.batch_size, 
                                     input_height=FLAGS.input_height, 
                                     input_width=FLAGS.input_width, 
                                     is_color=FLAGS.is_color)
    with tf.device('/gpu:0'):
      network = net_select(FLAGS.net_name)
      features = network(images)
      if FLAGS.flip_flag:
        features_flipped = network(tf.reverse(images, axis=[2]), reuse=True)
        features = (features+features_flipped)/2
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
    savemat(os.path.join(FLAGS.feature_dir, FLAGS.net_name+'_'+FLAGS.model_name, FLAGS.fea_name+'_split'+str(split)+'_'+step+'.mat'), {'wfea': wfea})

    print('Split %d Done.' % split)

    

def main(argv=None):
  tf.gfile.MakeDirs(os.path.join(FLAGS.feature_dir, FLAGS.net_name+'_'+FLAGS.model_name))
  for split in xrange(1, 11):
    evaluate(split)

if __name__ == '__main__':
  tf.app.run()
