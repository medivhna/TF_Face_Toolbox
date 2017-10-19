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

import tensorflow as tf

def softmax_cross_entropy_with_logits(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, 
                             name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    
  tf.add_to_collection('losses', cross_entropy_mean)

  return cross_entropy_mean

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

def triplet_loss(anchor, positive, negative, alpha):
  """Calculate the triplet loss according to the FaceNet paper
    
  Args:
    anchor: the embeddings for the anchor images.
    positive: the embeddings for the positive images.
    negative: the embeddings for the negative images.
  
  Returns:
    the triplet loss according to the FaceNet paper as a float tensor.
  """
  with tf.variable_scope('triplet_loss'):
      pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
      neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
      basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)

      triplet_loss_mean = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

      tf.add_to_collection('losses', triplet_loss_mean)
      
  return loss