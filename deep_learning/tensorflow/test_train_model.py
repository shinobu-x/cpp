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

  def _add_rnn_layers(convolved, lengths):
    if params.cell_type != "cudnn_lstm":
      outputs = _add_regular_rnn_layers(
        convolved,
        lengths)
    else:
      outputs = _add_cudnn_rnn_layers(convolved)

    mask = tf.tile(
      tf.expand_dims(
        tf.sequence_mask(
          lengths,
          tf.shape(
            outputs)([1]),
        2),
      [1, 1, tf.shape(outputs)[2]])

    zero_outside = tf.where(
      mask,
      outputs,
      tf.zeros_like(outputs))

    outputs = tf.reduce_sum(
      zero_outside,
      axis = 1)

  def _add_fc_layers(final_state):
    return tf.layers.dense(
      final_state,
      params.num_classes)

  inks, lengths, labels = _get_input_tensors(features, labels)
  convolved, lengths = _add_conv_layers(inks, lengths)
  final_size = _add_rnn_layers(convolved, lengths)
  logists = _add_fc_layers(final_state)

  cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logist(
      labels = labels,
      logists = logists))

  train_op = tf.contrib.layers.optimize_loss(
    loss = cross_entropy,
    global_step = tf.train.get_global_step(),
    learning_rate = params.training_rate,
    optimizer = "Adam",
    clip_gradients = params.gradient_clipping_norm,
    summaries = ["learning_rate", "loss", "gradients", "gradient_norm"])

  predictions = tf.argmax(logists, axis = 1)

  return tf.estimator.EstimatorSpec(
    mode = mode,
    predictions = {"logists": logists, "predictions": predictions},
    loss = cross_entropy,
    train_op = train_op,
    eval_metric_ops = {"accuracy": tf.matrics.accuracy(labels, predictions)})

def create_estimator_and_specs(run_config):
  model_params = tf.contrib.training.HParams(
    num_layers = FLAGS.sum_layers,
    num_nodes = FLAGS.num_nodes,
    batch_size = FLAGS.batch_size,
    num_conv = ast.literal_eval(FLAGS.num_conv),
    conv_len = ast.literal_eval(FLAGS.conv_len),
    num_classes = get_num_classes(),
    learning_rate = FLAGS.learning_rate,
    gradient_clipping_norm = FLAGS.gradient_clipping_norm,
    cell_type = FLAGS.cell_type,
    batch_norm = FLAGS.batch_norm,
    dropout = FLAGS.dropout)

  estimator = tf.estimator.Estimator(
    model_fn = model_fn,
    config = run_config,
    params = model_params)

  train_spec = tf.estimator.TrainSpec(
    input_fn = get_input_fn(
      mode = tf.estimator.ModeKeys.TRAIN,
      tfrecord_pattern = FLAGS.training_data,
      batch_size = FLAGS.batch_size),
    max_steps = FLAGS.steps)

  eval_spec = tf.estimator.EvalSpec(
    input_fn = get_input_fn(
      mode = tf.estimator.ModeKeys.EVAL,
      tfrecord_pattern = FLAGS.eval_data,
      batch_size = FLAGS.batch_size))

  return estimator, train_spec, eval_spec

def main(unused_args):
  estimator, train_spec, eval_spec = create_estimator_and_specs(
    run_config = tf.estimator.RunConfig(
      model_dir = FLAGS.model_dir,
      save_checkpoints_secs = 300,
      save_summary_steps = 100))

  tf.estimator.train_and_evaluate(
    estimator,
    train_spec,
    eval_spec)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    "type",
    "bool",
    lambda v: v.lower() == "true")

  parser.add_argument(
    "--training_data",
    type = str,
    default = "",
    help = "Path to training data (tf.Example in TFRecord format).")

  parser.add_argument(
    "--eval_data",
    type = str,
    default = "",
    help = "Path to evaluation data (tf.Example in TFRecord format).")

  parser.add_argument(
    "--classes_file",
    type = str,
    default = "",
    help = "Path to a file with the classes - one class per line.")

  parser.add_argument(
    "--num_layers",
    type = int,
    default = 3,
    help = "Number of recurrent nural network layers.")

  parser.add_argument(
    "--num_nodes",
    type = int,
    default = 128,
    help = "Number of node per recurrent network layer.")

  parser.add_argument(
    "--conv_len",
    type = str,
    default = "[5, 5, 3]",
    help = "Length of the convolution filters.")

  parser.add_argument(
    "--cell_type",
    type = str,
    default = "lstm",
    help = "Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")

  parser.add_argument(
    "--batch_norm",
    type = "bool",
    default = "False",
    help = "Whether to enable batch normalization or not.")

  parser.add_argument(
    "--learning_rate",
    type = float,
    default = 0.0001,
    help = "Learning rate used for training.")

  parser.add_argument(
    "--gradient_clipping_norm",
    type = float,
    default = 9.0,
    help = "Gradient clipping norm used during training.")

  parser.add_argument(
    "--dropout",
    type = float,
    default = 0.3,
    help = "Dropout used for convolutions and bidi lstm layers.")

  parser.add_argument(
    "--steps",
    type = int,
    default = 100000,
    help = "Number of training steps.")

  parser.add_argument(
    "--batch_size",
    type = int,
    default = 8,
    help = "Batch size to use for training and evaluations.")

  parser.add_argument(
    "--model_dir",
    type = str,
    default = "",
    help = "path to store the model checkpoints.")

  parser.add_argument(
    "--self_test",
    type = "bool",
    help = "Whther to enable batch normalization or not.")

  FLAGS, unparsed = parser.parser_known_args()
  tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)

