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

import copy
import os.path
import math
import random
import itertools
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
from scipy import misc

from preprocessing import data_augmentation

# get methods
def get_image_paths(list_path):
  image_paths_flat = []
  for line in open(os.path.expanduser(list_path), 'r'):
    part_line = line.split()
    image_path = part_line[0]
    image_paths_flat.append(image_path)

  num_examples_total = len(image_paths_flat)

  return image_paths_flat, num_examples_total


# get methods
def get_image_paths_and_labels(list_path):
  image_paths_flat = []
  labels_flat = []
  for line in open(os.path.expanduser(list_path), 'r'):
    part_line = line.split()
    image_path = part_line[0]
    label_index = part_line[1]
    image_paths_flat.append(image_path)
    labels_flat.append(int(label_index))

  num_examples_total = len(labels_flat)
  num_classes_total = max(labels_flat)+1

  return image_paths_flat, labels_flat, num_examples_total, num_classes_total

def get_image_paths_and_labels_and_cam(list_path):
  image_paths_flat = []
  labels_flat = []
  cam_flat = []
  for line in open(os.path.expanduser(list_path), 'r'):
    part_line = line.split()
    image_path = part_line[0]
    label_index = part_line[1]
    cam_index = part_line[2]
    image_paths_flat.append(image_path)
    labels_flat.append(int(label_index))
    cam_flat.append(int(cam_index))

  num_examples_total = len(labels_flat)
  num_classes_total = max(labels_flat)+1
  num_cameras_total = max(cam_flat)+1

  return image_paths_flat, labels_flat, cam_flat, num_examples_total, num_classes_total, num_cameras_total

def get_image_paths_and_labels_dict(list_path, num_per_class):
  image_path_list_tmp = []
  image_path_list = []
  for line in open(os.path.expanduser(list_path), 'r'):
    part_line = line.split(' ')
    image_path = part_line[0]
    label_index = part_line[1]
    if len(image_path_list_tmp) <= int(label_index):
      image_path_list_tmp.append([])
    image_path_list_tmp[int(label_index)].append(image_path)

  num_examples_total = 0
  for idx in range(len(image_path_list_tmp)):
    if len(image_path_list_tmp[idx]) >= num_per_class:
      image_path_list.append(image_path_list_tmp[idx])
      num_examples_total += len(image_path_list_tmp[idx])

  num_classes_total = len(image_path_list)

  return image_path_list, num_examples_total, num_classes_total

def get_image_paths_and_labels_cam_dict(list_path, num_per_class):
  image_path_list_tmp = []
  image_path_list = []
  num_cameras_total = -1
  for line in open(os.path.expanduser(list_path), 'r'):
    part_line = line.split(' ')
    image_path = part_line[0]
    label_index = part_line[1]
    cam_index = part_line[2]
    while len(image_path_list_tmp) <= int(label_index):
      image_path_list_tmp.append([])
    image_path_list_tmp[int(label_index)].append((image_path, int(cam_index)))
    if int(cam_index) > num_cameras_total:
      num_cameras_total = int(cam_index)

  num_examples_total = 0
  for idx in range(len(image_path_list_tmp)):
    if len(image_path_list_tmp[idx]) >= num_per_class:
      image_path_list.append(image_path_list_tmp[idx])
      num_examples_total += len(image_path_list_tmp[idx])

  num_classes_total = len(image_path_list)
  num_cameras_total += 1

  return image_path_list, num_examples_total, num_classes_total, num_cameras_total

def get_image_paths_and_labels_cam_dict_self(list_path, labels, num_per_class):
  image_path_list_tmp = []
  image_path_list = []
  num_cameras_total = -1
  for idx, line in enumerate(open(os.path.expanduser(list_path), 'r')):
    if labels[idx] == -1:
      continue
    part_line = line.split(' ')
    image_path = part_line[0]
    #label_index = part_line[1]
    label_index = labels[idx]
    cam_index = part_line[2]
    while len(image_path_list_tmp) <= int(label_index):
      image_path_list_tmp.append([])
    image_path_list_tmp[int(label_index)].append((image_path, int(cam_index)))
    if int(cam_index) > num_cameras_total:
      num_cameras_total = int(cam_index)

  num_examples_total = 0
  for idx in range(len(image_path_list_tmp)):
    if len(image_path_list_tmp[idx]) >= num_per_class:
      image_path_list.append(image_path_list_tmp[idx])
      num_examples_total += len(image_path_list_tmp[idx])

  num_classes_total = len(image_path_list)
  num_cameras_total += 1

  return image_path_list, num_examples_total, num_classes_total, num_cameras_total

def eval_inputs(data_list_path, 
                batch_size, 
                is_color,
                input_height, 
                input_width):

  def _parse_function(filename):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=num_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [input_height, input_width])
    image = (image - 0.5) / 0.5

    return image

  def _gen():
    for image in image_list:
      yield image

  num_channels = 3 if is_color else 1
  image_mean = tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1, 1, num_channels])
  image_std = tf.reshape(tf.constant([0.229, 0.224, 0.225]), [1, 1, num_channels])
  image_list, num_examples_total = get_image_paths(data_list_path)
  num_examples = len(image_list)
  print('%d images loaded' % (num_examples))

  dataset = tf.data.Dataset.from_generator(_gen, (tf.string))
  dataset = dataset.repeat()
  # Temporary implementation
  dataset = dataset.apply(tf.contrib.data.map_and_batch(_parse_function, batch_size,
                                                        num_parallel_calls=8))
#                                                        num_parallel_batches=4,  # dataset = dataset.map(, num_parallel_calls=16)
  # dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1)
  image_batch = dataset.make_one_shot_iterator().get_next()

  image_batch.set_shape((batch_size, input_height, input_width, num_channels))

  return image_batch, num_examples


# input methods
def train_inputs(data_list_path, 
                 input_height, 
                 input_width, 
                 crop_height=-1,
                 crop_width=-1,
                 is_color=1,
                 augmentation=0,
                 batch_size=-1,
                 num_classes=-1,
                 num_per_class=-1):

  def _parse_function(filename, label):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=num_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resize (crop_size is None.) or random crop (crop_size is given.)
    image = tf.image.resize_images(image, [input_height, input_width])
    if crop_height != -1 and crop_width != -1:
      image = tf.random_crop(image, [crop_height, crop_width, num_channels])

    # Data augmentation
    if augmentation:
      image = data_augmentation(image)
    else:
      image = tf.image.random_flip_left_right(image)
    image = (image - 0.5) / 0.5

    return image, label

  def _gen_random():
    rand_idx = np.random.permutation(num_examples_total)
    for idx in rand_idx:
      yield (image_list[idx], label_list[idx])

  def _gen_balance():
    image_label_dict_new = copy.deepcopy(image_label_dict)
    for idx in range(len(image_label_dict_new)):
      np.random.shuffle(image_label_dict_new[idx])
    while 1:
      rand_idx_class = np.random.choice(range(len(image_label_dict_new)), num_classes)
      for idx in rand_idx_class:
        if len(image_label_dict_new[idx])<num_per_class:
          image_label_dict_new[idx] = copy.deepcopy(image_label_dict[idx])
          np.random.shuffle(image_label_dict_new[idx])
        for _ in range(num_per_class):
          path = image_label_dict_new[idx].pop()
          yield (path, idx)

  num_channels = 3 if is_color else 1
  image_mean = tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1, 1, num_channels])
  image_std = tf.reshape(tf.constant([0.229, 0.224, 0.225]), [1, 1, num_channels])

  if batch_size == -1:
    assert num_classes != -1 and num_per_class != -1
    batch_size = num_classes*num_per_class

    _gen = _gen_balance
    image_label_dict, num_examples_total, num_classes_total = get_image_paths_and_labels_dict(data_list_path, num_per_class)
  else:
    _gen = _gen_random
    image_list, label_list, num_examples_total, num_classes_total = get_image_paths_and_labels(data_list_path)
  print('%d images loaded, totally %d classes' % (num_examples_total, num_classes_total))

  dataset = tf.data.Dataset.from_generator(_gen, (tf.string, tf.int32))
  dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count()/2)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1)
  image_batch, label_batch = dataset.make_one_shot_iterator().get_next()

  if crop_height != -1 and crop_width != -1:
    input_height = crop_height
    input_width = crop_width

  image_batch.set_shape((batch_size, input_height, input_width, num_channels))
  label_batch.set_shape((batch_size))

  tf.summary.image('images', image_batch, max_outputs=12)

  outputs = {}
  outputs['images'] = image_batch
  outputs['labels'] = label_batch
  outputs['num_classes'] = num_classes_total
  outputs['num_examples'] = num_examples_total
  
  return outputs


def train_inputs_self(data_list_path, 
                     labels, 
                     input_height, 
                     input_width, 
                     crop_height=-1,
                     crop_width=-1,
                     is_color=1,
                     augmentation=0,
                     batch_size=-1,
                     num_classes=-1,
                     num_per_class=-1):

  def _parse_function(filename, label, cam_id):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=num_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resize (crop_size is None.) or random crop (crop_size is given.)
    image = tf.image.resize_images(image, [input_height, input_width])
    if crop_height != -1 and crop_width != -1:
      image = tf.random_crop(image, [crop_height, crop_width, num_channels])

    # Data augmentation
    if augmentation:
      image = data_augmentation(image)
    else:
      image = tf.image.random_flip_left_right(image)

    image = (image - 0.5) / 0.5
#(image-image_mean)/image_std

    return image, label, cam_id

  def _gen_random():
    rand_idx = np.random.permutation(num_examples_total)
    for idx in rand_idx:
      if label_list[idx] != -1:
        yield (image_list[idx], label_list[idx], cam_id_list[idx])

  def _gen_balance():
    image_label_dict_new = copy.deepcopy(image_label_dict)
    for idx in range(len(image_label_dict_new)):
      np.random.shuffle(image_label_dict_new[idx])
    while 1:
      rand_idx_class = np.random.choice(range(len(image_label_dict_new)), num_classes, replace=False)
      for idx in rand_idx_class:
        if len(image_label_dict_new[idx])<num_per_class:
          image_label_dict_new[idx] = copy.deepcopy(image_label_dict[idx])
          np.random.shuffle(image_label_dict_new[idx])
        for _ in range(num_per_class):
          [path, cam_id] = image_label_dict_new[idx].pop()
          yield (path, idx, cam_id)

  num_channels = 3 if is_color else 1
  image_mean = tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1, 1, num_channels])
  image_std = tf.reshape(tf.constant([0.229, 0.224, 0.225]), [1, 1, num_channels])

  if batch_size == -1:
    assert num_classes != -1 and num_per_class != -1
    batch_size = num_classes*num_per_class

    _gen = _gen_balance
    image_label_dict, num_examples_total, num_classes_total, num_cameras_total = get_image_paths_and_labels_cam_dict_self(data_list_path, labels, num_per_class)
  else:
    _gen = _gen_random
    image_list, label_list, cam_id_list, num_examples_total, num_classes_total, num_cameras_total = get_image_paths_and_labels_and_cam(data_list_path)
    label_list = labels
  print('%d images loaded, totally %d classes, from %d cameras' % (num_examples_total, num_classes_total, num_cameras_total))

  dataset = tf.data.Dataset.from_generator(_gen, (tf.string, tf.int32, tf.int32))
  dataset = dataset.repeat()
  # Temporary implementation
  dataset = dataset.apply(tf.contrib.data.map_and_batch(_parse_function, batch_size,
                                                        num_parallel_calls=8))
  # dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count()/2)
  # dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(batch_size)
  image_batch, label_batch, cam_id_batch = dataset.make_one_shot_iterator().get_next()

  if crop_height != -1 and crop_width != -1:
    input_height = crop_height
    input_width = crop_width

  image_batch.set_shape((batch_size, input_height, input_width, num_channels))
  label_batch.set_shape((batch_size))
  cam_id_batch.set_shape((batch_size))

  tf.summary.image('images', image_batch, max_outputs=12)

  outputs = {}
  outputs['images'] = image_batch
  outputs['labels'] = label_batch
  outputs['cam_ids'] = cam_id_batch
  outputs['num_classes'] = num_classes_total
  outputs['num_cameras'] = num_cameras_total
  outputs['num_examples'] = num_examples_total
  
  return outputs
