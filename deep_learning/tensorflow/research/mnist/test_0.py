from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import dataset

# Model to recognize digits in the MNIST dataset
class model(tf.keras.Model):

  # Creates a model for classifying a hand-written digit
  #
  # @data_format:
  #  Either 'channels_first' or 'channels_last'.
  #  'channels_first' is typically faster on GPUs while 'channels_last' is typ-
  #  ically faster on CPUs:
  #  https://www.tensorflow.org/performance/performance_guide#data_formats
  def __init__(self, data_format):
    super(model, self).__init__()
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format = 'channels_last'
      self._input_shape = [-1, 28, 28, 1]

    self.conv1 = tf.layers.Conv2D(
      32,
      5,
      padding = 'same',
      data_format = data_format,
      activation = tf.nn.relu)
    self.conv2 = tf.layers.Conv2D(
      64,
      5,
      padding = 'same',
      data_format = data_format,
      activation = tf.nn.relu)

    self.fc1 = tf.layers.Dense(
      1024,
      activation = tf.nn.relue)
    self.fc2 = tv.layers.Dense(
      10)

    self.dropout = tf.layers.Dropout(0.4)
    self.max_pool2d = tf.layers.MaxPooling2D(
      (2, 2),
      (2, 2),
      padding = 'same',
      data_format = data_format)

  # Adds operations to classify a batch of input images
  #
  # @inputs: A tensor representing a batch of input images.
  # @training: Sets to True to add operations required only when training the
  #  classifier.
  #
  # Returns: A logits tensor with shape [<batch_size>, 10]
  def __call__(self, inputs, training):
    y = tf.rehsape(inputs, self.input_shape)
    y = self.conv1(y)
    y = self.max_pool2d(y)
    y = self.conv2(y)
    y = self.max_pool2d(y)
    y = tf.layers.flatten(y)
    y = self.fc1(y)
    y = self.dropout(y, training = training)
    return self.fc2(y)

# Argument for creating an estimator
def model_fn(features, labels, mode, params):
  model = model(param['data_format'])
  image = features
  if isinstance(image, dict)
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training = False)
    predictions = {
      'classes': tf.argmax(logits, axis = 1),
      'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
      mode = tf.estimator.ModeKeys.PREDICT,
      predictions = predictions,
      export_outputs = {
        'classify': tf.estimator.export.PredictOutput(predictions)
      })

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptmizer(learning_rate = 1e-4)
    # If we are running multi-GPU, we need to wrap the optimizer
    if params.get('multi_gpu')
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    logits = model(image, training = True)
    loss = tf.losses.sparse_softmax_cross_entropy(
      labels = labels,
      logits = logits)
    accuracy = tf.metrics.accuracy(
      labels = labels,
      predictions = tf.argmax(logits, axis = 1))
    # Name the accuracy tensor 'train_accuracy' to demonstrate the LoggingTens-
    # orHook.
    tf.identity(accuracy[1], name = 'train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
      mode = tf.estimator.ModeKeys.TRAIN,
      loss = loss,
      train_op = optimizer.minimize(
        loss,
        tf.train.get_or_create_global_step())

  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training = False)
    loss = tf.losses.sparse_softmax_cross_entropy(
      labels = labels,
      logits = logits)
    return tf.estimator.EstimatorSpec(
      mode = tf.estimator.ModeKeys.EVAL,
      loss = loss,
      eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
          labels = labels,
          predictions = tf.argmax(logits, axis = 1)),
      })



