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

import os.path
import math
import threading

import numpy as np
import tensorflow as tf
from scipy import misc

from preprocessing import data_augment

def get_image_paths_and_labels(list_path):
  image_paths_flat = []
  labels_flat = []
  for line in open(os.path.expanduser(list_path), 'r'):
    part_line = line.split(' ')
    image_path = part_line[0]
    label_index = part_line[1]
    image_paths_flat.append(image_path)
    labels_flat.append(int(label_index))

  return image_paths_flat, labels_flat

def train_inputs(data_list_path, 
                 batch_size, 
                 is_color, 
                 input_height, 
                 input_width, 
                 crop_height=None,
                 crop_width=None,
                 augment=False,
                 num_buffer=5,
                 num_preprocess_threads=None):

  def _parse_function(filename, label):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=num_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resize (crop_size is None.) or random crop (crop_size is given.)
    if crop_height is None or crop_width is None:
      image = tf.image.resize_images(image, [input_height, input_width])
    else:
      image = tf.random_crop(image, [crop_height, crop_width, num_channels])

    # Data augmentation
    if augment:
      image = data_augment(image)
    else:
      image = tf.image.random_flip_left_right(image)

    image = tf.image.per_image_standardization(image)

    return image, label

  def _gen():
    rand_idx = np.random.permutation(num_examples)
    for idx in rand_idx:
      yield (image_list[idx], label_list[idx])

  num_channels = 3 if is_color else 1
  image_list, label_list = get_image_paths_and_labels(data_list_path)
  num_examples = len(label_list)
  num_classes = max(label_list)+1
  print('%d images loaded, totally %d classes' % (num_examples, num_classes))

  dataset = tf.data.Dataset.from_generator(_gen, (tf.string, tf.int32))
  dataset = dataset.map(_parse_function, num_parallel_calls=num_preprocess_threads)
  dataset = dataset.prefetch(buffer_size=num_buffer*batch_size)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  image_batch, label_batch = dataset.make_one_shot_iterator().get_next()

  if crop_height is not None and crop_width is not None:
    input_height = crop_height
    input_width = crop_width
  image_batch.set_shape((batch_size, input_height, input_width, num_channels))

  tf.summary.image('images', image_batch, max_outputs=10)
  
  return image_batch, label_batch, num_classes, num_examples

def eval_inputs(data_list_path, 
                batch_size, 
                is_color,
                input_height, 
                input_width,
                num_buffer=16):

  def _parse_function(filename):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=num_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [input_height, input_width])
    image = tf.image.per_image_standardization(image)
    image.set_shape((input_height, input_width, num_channels))

    return image

  def _gen():
    for image in image_list:
      yield image

  num_channels = 3 if is_color else 1
  image_list = []
  for line in open(os.path.expanduser(data_list_path), 'r'):
    part_line = line.strip().split(' ')
    image_list.append(part_line[0])
  num_examples = len(image_list)
  print('%d images loaded' % num_examples)

  dataset = tf.data.Dataset.from_generator(_gen, tf.string)
  dataset = dataset.map(_parse_function)
  dataset = dataset.prefetch(buffer_size=num_buffer*batch_size)
  dataset = dataset.batch(batch_size)
  image_batch = dataset.make_one_shot_iterator().get_next()

  return image_batch, num_examples