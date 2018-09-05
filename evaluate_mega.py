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
from data import eval_inputs_megaface
from nets.factory import net_select

parser = argparse.ArgumentParser()
# Name configures
parser.add_argument('--net_name', type=str,
                    help='Name of the network architecture.')
parser.add_argument('--model_name', type=str,
                    help='Name of the training model.')
parser.add_argument('--fea_name', type=str,
                    help='Prefix name of feature files.')
# Directory configures
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
parser.add_argument('--data_list_path', type=str, default='~/work/tf_face_new/verify/megaface_devkit/template_lists'
                    help='Path to the list of testing data.')
# Hyperparameters configures
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of images to process in a batch.')
FLAGS = parser.parse_args()


def evaluate(template_name):
  """Eval deep network for a number of steps."""

  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    images, names, num_images = eval_inputs_megaface(os.path.join(FLAGS.data_list_path, template_name), 
                                                                 batch_size=1, 
                                                                 input_height=FLAGS.input_height, 
                                                                 input_width=FLAGS.input_width, 
                                                                 is_color=FLAGS.is_color)
    with tf.device('/gpu:0'):
      network = net_select(FLAGS.net_name)
      features = network(images)
      if FLAGS.flip_flag:
        features_flipped = network(tf.reverse(images, axis=[2]), reuse=True)
        features = tf.concat([features, features_flipped], axis=1)#()/2
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

    iters = 0
    while iters < num_images:
      start_time = time.time()
      fea, name = sess.run(features, names)
      iters += 1
      matio(fea, name)
      print('%d/%d features extracted... %.2fms elapsed'% iters)

    # Only save num_images features
    wfea = wfea[0:num_images, :]
    print('Totally extracted %d features.'%(wfea.shape[0]))

    print('Done.')

    

def main(argv=None):
  tf.gfile.MakeDirs(os.path.join(FLAGS.feature_dir, FLAGS.net_name+'_'+FLAGS.model_name, 'megaface'))
  evaluate('facescrub_features_list.json')
  for distractor_size in [10, 100, 1000, 10000, 100000, 1000000]:
    evaluate('megaface_features_list.json_'+str(distractor_size)+'_1')

if __name__ == '__main__':
  tf.app.run()
