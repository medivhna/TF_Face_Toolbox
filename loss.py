# Copyright 2018 Guanshuo Wang. All Rights Reserved.
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

import tensorflow as tf

def focal_loss(logits, labels, gamma=1.0, alpha=2.0):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  nd_indices = tf.stack([tf.range(logits.get_shape().as_list()[0], dtype=tf.int32), labels], axis=1)
  scores = tf.gather_nd(tf.nn.softmax(logits), nd_indices)

  focal_entropy = gamma*tf.pow(1-scores, alpha)*cross_entropy
  loss = tf.reduce_mean(focal_entropy, name='focal_entropy')
  tf.add_to_collection('losses', loss)
      
  return loss

def center_loss(features, labels, num_classes, alpha=0.99, weight=1.0):
  """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
     (http://ydwen.github.io/papers/WenECCV16.pdf)
  """
  num_features = features.get_shape()[1]
  centers = tf.get_variable('centers', [num_classes, num_features], dtype=tf.float32,
                            initializer=tf.constant_initializer(0), trainable=False)
  labels = tf.reshape(labels, [-1])
  centers_batch = tf.gather(centers, labels)
  diffs = (1 - alpha) * (centers_batch - features)
  centers = tf.scatter_sub(centers, labels, diffs)
  
  center_loss_mean = tf.reduce_mean(tf.square(features - centers_batch))

  tf.add_to_collection('losses', weight*center_loss_mean)
  
  return center_loss_mean, centers

def batch_hard_triplet_loss(features, labels, margin=None, metric='euclidean'):
  def all_diffs(a, b):
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

  def cdist(a, b, metric='euclidean'):
    with tf.name_scope("cdist"):
      diffs = all_diffs(a, b)
      if metric == 'sqeuclidean':
        return tf.reduce_sum(tf.square(diffs), axis=-1)
      elif metric == 'euclidean':
        return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
      elif metric == 'cityblock':
        return tf.reduce_sum(tf.abs(diffs), axis=-1)
      else:
        raise NotImplementedError(
          'The following metric is not implemented by `cdist` yet: {}'.format(metric))

  with tf.name_scope("batch_hard"):
    dists = cdist(features, features)
    intra_mask = tf.equal(tf.expand_dims(labels, axis=1), tf.expand_dims(labels, axis=0))

    neg_mask = tf.cast(tf.logical_not(intra_mask), dtype=tf.float32)
    pos_mask = tf.cast(tf.logical_xor(intra_mask, tf.eye(tf.shape(labels)[0], dtype=tf.bool)), dtype=tf.float32)

    hardest_pos = tf.reduce_max(dists*pos_mask, axis=1)
    hardest_neg = tf.reduce_min(dists*neg_mask+1e6*tf.cast(intra_mask, dtype=tf.float32), axis=1)

    if margin is None:
      diff = tf.nn.softplus(hardest_pos-hardest_neg)
    else:
      diff = tf.maximum(0.0, hardest_pos-hardest_neg+margin)

  return diff