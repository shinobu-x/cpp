from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import dataset

# Model to recognize digits in the MNIST dataset
class Model(tf.keras.Model):

  # Creates a model for classifying a hand-written digit
  #
  # @data_format:
  #  Either 'channels_first' or 'channels_last'.
  #  'channels_first' is typically faster on GPUs while 'channels_last' is typ-
  #  ically faster on CPUs:
  #  https://www.tensorflow.org/performance/performance_guide#data_formats
  def __init__(self, data_format):
    super(Model, self).__init__()
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format == 'channels_last'
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
  model = Model(params['data_format'])
  image = features
  if isinstance(image, dict):
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
    if params.get('multi_gpu'):
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
        tf.train.get_or_create_global_step()))

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

def validate_batch_size_for_multi_gpu(batch_size):
  # With multi GPUs, batch-size must be a multiple of the number of available
  # GPUs.
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError(
      'Multi-GPU mode was specifed, but no GPUs were found. '
      'To use CPU, run without --multi_gpu.')
  remainder = batch_size % num_gpus
  if reaminder:
    err = (
      'With multiple GPUSs, batch-size must be a multiple of the number of '
      'available of the number of available GPUs. '
      'Found {} GPUs with a batch size of {}: '
      'Try --batch_size = {} instead.')
    raise ValueError(err)

def main(unused_argv):
  model_function = model_fn
  if FLAGS.multi_gpu:
    validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    # There are two steps required if using multiple GPUs:
    #  1.Wraps the model_fn.
    #  2.Wraps the optimizer.
    # The first happens here, and the second happens in the model_fn itself wh-
    # en the optimizer is defined.
    model_function = tf.contrib.estimator.replicate_model_fn(
      model_fn,
      loss_reduction = tf.losses.Reduction.MEAN)
  data_format = FLAGS.data_format
  if data_format is None:
    data_format = (
      'channels_first'
      if tf.test.is_built_with_cuda() else 'channels_last')
  mnist_classifier = tf.estimator.Estimator(
    model_fn = model_function,
    model_dir = FLAGS.model_dir,
    params = {
      'data_format': data_format,
      'multi_gpu': FLAGS.multi_gpu
    }
  )
  # Trains the model
  def train_input_fn():
    # Choosing shuffle buffer sizes, larger sizes result in better randomness,
    # while smaller sizes use less memory. MNIST is a small enough dataset that
    # we can easily shuffle the full epoch
    ds = dataset.train(FLAGS.data_dir)
    ds = ds.cache().shuffle(
      buffer_size = 50000).batch(
        FLAGS.batch_size).repeat(
          FLAGS.train_epochs)
    return ds

  # Sets up training hook that logs the training accuracy every 100 steps.
  tensors_to_log = {'train_accuracy': 'train_accuracy'}
  logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log,
    every_n_iter = 100)
  mnist_classifier.train(
    input_fn = train_input_fn,
    hooks = [logging_hook])

  # Evaluates the model and print results
  def eval_input_fn():
    return dataset.test(FLAGS.data_dir).batch(
      FLAGS.batch_size).make_one_shot_iterator().get_next()
  eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
  print()
  print('Evaluation results:\n\t%s' % eval_results)

  # Exports the model
  if FLAGS.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
      {'image': image,
    })
  mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)

class MNISTArgParser(argparse.ArgumentParser):
  def __init__(self):
    super(MNISTArgParser, self).__init__()

    self.add_argument(
      '--multi_gpu', action = 'store_true',
      help = 'Run accross all available GPUs.')
    self.add_argument(
      '--batch_size', type = int, default = 100,
      help = 'Number of images to process in a batch.')
    self.add_argument(
      '--data_dir', type = str, default = '/tmp/mnist_data',
      help = 'Path to directory containing the MNIST dataset')
    self.add_argument(
      '--model_dir', type = str, default = '/tmp/mnist_model',
      help = 'Path to directory storing the model')
    self.add_argument(
      '--train_epochs', type = int, default = 40,
      help = 'Number of epochs to train')
    self.add_argument(
      '--data_format', type = str, default = None,
      choices = ['channels_first', 'channels_last'],
      help = 'A flag to override the data format used in the models. '
        'channels_first provides a performance boost on GPU but is not always '
        'compatible with CPU. If left unspecified, the data format will be '
        'chosen automatically based on whether tensorflow was build for CPU '
        'or GPU.')
    self.add_argument(
      '--expor_dir', type = str,
      help = 'Path to directory storing the exported saved model')

if __name__ == '__main__':
  parser = MNISTArgParser()
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
