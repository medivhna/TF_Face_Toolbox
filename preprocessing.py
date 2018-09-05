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

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import transform

def data_augmentation(image):
  num_channels = image.get_shape().as_list()[-1]
  image = tf.image.random_flip_left_right(image)
  delta = tf.random_uniform(shape=[], minval=0, maxval=0.2)
  image = tf.cond(tf.less(delta, 0.1),
                  lambda: tf.image.adjust_brightness(image, -delta),
                  lambda: image)
  if num_channels == 3:
    delta = tf.random_uniform(shape=[], minval=0, maxval=0.4)
    image = tf.cond(tf.less(delta, 0.2),
                    lambda: tf.image.adjust_hue(image, -delta),
                    lambda: image)    
    delta = tf.random_uniform(shape=[], minval=0.6, maxval=1.4)
    image = tf.cond(tf.less(delta, 1.0),
                    lambda: tf.image.adjust_saturation(image, delta),
                    lambda: image)
  return image


def _random_zoom_in_out(image, rand_rate=0.5):
  assert rand_rate <= 1.0 and rand_rate >= 0.5
  scale = tf.random_uniform([], 1.0-rand_rate, 2.0-rand_rate)
  scale = tf.minimum(scale, 1.0)
  source_shape = tf.cast(image.get_shape()[0:2], dtype=tf.int32)
  target_shape = tf.cast(scale*tf.cast(source_shape, dtype=tf.float32), dtype=tf.int32)

  image = tf.image.resize_images(image, target_shape)
  image = tf.image.resize_images(image, source_shape)

  return image

def _random_affine_distort(image):
  source_x = np.array([38, 89, 64])
  source_y = np.array([55, 55, 105])
  rnd = random.randint(0, 728)
  target_x = np.array([source_x[0] + rnd/243-1, 
                       source_x[1] + rnd%81/27-1, 
                       source_x[2] + rnd%9/3-1])
  target_y = np.array([source_y[0] + rnd%243/81-1,
                       source_y[1] + rnd%27/9-1,
                       source_y[2] + rnd%3-1])

  A = np.vstack((source_x, source_y, np.ones(3)))
  A = np.transpose(A)
  tform_x = np.linalg.solve(A, target_x)
  tform_y = np.linalg.solve(A, target_y)
  tform = tform_x.tolist() + tform_y.tolist() + [0, 0]
  image = transform(image, tform, interpolation='BILINEAR')

  return image

