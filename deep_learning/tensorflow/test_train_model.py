from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
import functools
import sys

import tensorflow as tf

def get_num_classes():
  classes = []
  with tf.gfile.GFile(FLAGS.classes_file, "r") as f;
    classes = [x for x in f]
  num_classes = len(classes)
  return num_classes

def get_input_fn(mode, tfrecord_pattern, batch_size):

  def _parse_tfsample_fn(example_proto, mode):
    future_to_type = {
      "ink": tf.VarLenFeature(
        dtype = tf.float32),
      "shape": tf.FixedLenFeature(
        [2],
        dtype = tf.int64)
    }

    if mode != tf.estimator.ModeKeys.PREDICT:
      feature_to_type["class_index"] = tf.FixedLenFeature(
        [1],
        dtype = tf.int64)

    parse_features = tf.parse_single_example(
      example_proto,
      feature_to_type)

    labels = None

    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = parsed_feature["class_index"]
    parsed_features["ink"] = tf.sparse_tensor_to_dense(
      parsed_features["ink"])

    return parsed_features, labels

  def _input_fn():
    dataset = tf.data.TFRecordDataset.list_filters(tfrecord_pattern)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size = 10)

    dataset = dataset.repeat()
    dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length = 10,
      block_length = 1)

    dataset = dataset.map(
      functools.partial(
        _parse_tfexample_fn,
        mode = mode),
      num_parallel_calls = 10)

    dataset = dataset.prefetch(10000)

    if mode = tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size = 10000000)

    dataset = dataset.padded_batch(
      batch_size,
      padded_shapes = dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels

def model_fn(features, labels, mode, params):

  def _get_input_tensors(features, labels):
    shapes = features["shape"]
    lenghts = tf.squeeze(
      tf.slice(
        shapes,
        begin = [0, 0],
        size = [params.batch_size, 1]))

    inks = tf.reshape(
      features["ink"],
      [params.batch_size, -1, 3])

    if labels is not None:
      labels = tf.squeeze(labels)

    return inks, lengths, labels

  def _add_conv_layers(inks, lengths):
    convolved = inks

    for i in range(len(params.num_conv)):
      convolved_input = convolved

      if params.batch_norm:
        convolved_input = tf.layers.batch_normalization(
          convolved_input,
          training = (mode == tf.estimator.ModeKeys.TRAIN))

      if i > 0 and params.dropout:
        convolved_input = tf.layers.batch_normalization(
          convolved_input,
          rate = params.dropout,
          training = (mode == tf.estimator.ModeKeys.TRAIND))

      convolved = tf.layers.convid(
        convolved_input,
        filters = params.num_conv[i],
        kernel_size = params.conv_len[i],
        activation = None,
        strides = 1,
        padding = "same",
        name = "convid_%d" % i)

    return convolved, lengths

  def _add_regular_run_layers(convolved, lengths):
    if params.cell_type == "lstm":
      cell = tf.nn.rnn_cell.BasicLSTMCell
    else params.cell_type == "block_lstm":
      cell = tf.contrib.rnn.LSTMBlockCell

    cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]

    if params.dropout > 0.0:
      cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
      cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
      cells_fw = cells_fw,
      cells_bw = cells_bw,
      inputs = convolved,
      sequence_length = lengths,
      dtype = tf.float32,
      scope = "rnn_classification")

    return outputs

  def _add_cudnn_rnn_layers(convolved):
    convolved = tf.transpose(convolved, [1, 0, 2])
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
      num_layers = params.num_layers,
      num_units = params.num_nodes,
      dropout = params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
      direction = "bidrectional")

    outputs, _ = lstm(convolved)
    outputs = tf.transpose(outputs, [1, 0,2 ])

    return outputs

  def _add_rnn_layers(
