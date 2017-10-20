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

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import transform

NUM_ANGMENTATION_SUPPORT = 4

def data_augment(image):
  aug_num = np.random.randint(low=0, high=NUM_ANGMENTATION_SUPPORT)
  aug_queue = np.random.permutation(NUM_ANGMENTATION_SUPPORT)[0:aug_num]
  for aug_idx in aug_queue:
    if aug_idx == 0:
      image = tf.image.random_flip_left_right(image)
    elif aug_idx == 1:
      image = _random_zoom_in_out(image, min_rate=0.2)
    elif aug_idx == 2:
      if num_channels == 1:
        image = tf.image.random_brightness(image, 0.5)
      elif num_channels == 3:
        image = tf.image.random_hue(image, 0.25)
    elif aug_idx == 3:
      if num_channels == 1:
        image = tf.image.random_contrast(image, 0.2, 0.8)
      elif num_channels == 3:
        image = tf.image.random_saturation(image, 0.3, 0.8)

  return image

def _random_zoom_in_out(image, min_rate=0.5):
  source_size = image.get_shape().as_list()[1]
  min_size = int(min_rate*source_size)
  target_size = random.randint(min_size, source_size)

  image = tf.image.resize_images(image, [source_size, source_size])

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

