from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy as np
from six.moves import urllib
from six.moves import xrange # pylint: disable = redefined-builtin

import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/tmp'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# Size of the validation set
VALIDATION_SIZE = 5000
# Set to None for random seed
SEED = 66478
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
# Number of steps between evaluation
EVAL_FREQUENCY = 100

FLAGS = None

def data_type():
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32

def maybe_download(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)

    filepath = os.path.join(WORK_DIRECTORY, filename)

    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            SOURCE_URL + filename,
            filepath)

        with tf.gfile.GFile(filepath) as f:
            size = f.size()

        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath

def extract_data(filename, num_images):
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            IMAGE_SIZE *
            IMAGE_SIZE *
            num_images *
            NUM_CHANNELS)
        data = np.frombuffer(
            buf,
            dtype = np.uint8).astype(np.float32)
        data = (data - (PIXEL_OPEN / 2.0)) / PIXEL_DEPTH
        data = data.reshape(
            num_images,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(
            buf,
            dtype = np.uint8).astype(np.int64)
        return labels

def fake_data(num_images):
    data = np.ndarray(
        shape = (
            num_images,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS),
        dtype = np.float32)
    labels = np.zeros(
        shape = (
            num_images,),
        dtype = np.int64)

    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5

    return data, labels

def error_rate(predictions, labels):
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])

def main(_):
    if FLAGS.self_test:
        print('Running self-test')
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
        num_epochs = 1
    else:
      train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
      train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
      test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
      test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

      train_data = extract_data(train_data_filename, 60000)
      train_labels = extract_labels(train_labels_filename, 60000)
      test_data = extract_data(test_data_filename, 10000)
      test_labels = extract_labels(test_labels_filename, 10000)

      validation_data = train_data[:VALIDATION_SIZE, ...]
      validation_lables = train_lables[:VALIDATION_SIZE]
      train_data = train_data[VALIDATION_SIZE:, ...]
      train_labels = train_labels[VALIDATION_SIZE:]
      num_epochs = NUM_EPOCHS

    train_size = train_labels.shape[0]

    train_data_node = tf.placeholder(
        data_type(),
        shape = (
            BATCH_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS))
    train_labels_node = tf.placeholder(
        tf.int64,
        shape = (BATCH_SIZE,))
    eval_data = tf.placeholder(
        data_type(),
        shape = (
            EVAL_BATCH_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS))

    conv1_weights = tv.Variable(
        tf.truncated_normal(
            [5, 5, NUM_CHANNELS, 32],
            stddev = 0.1,
            seed = SEED,
            dtype = data_type()))
    fc1_weights = tf.Variable(
        tf.truncated_normal(
            [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
            stddev = 0.1,
            seed = SEED,
            dtype = data_type()))
    fc1_biases = tf.Variable(
        tf.constant(
            0.1,
            shape = [512],
            dtype = data_type()))
    fc2_weights = tf.Variable(
        tf.truncated_normal(
            [512, NUM_LABELS],
            stddev = 0.1,
            seed = SEED,
            dtype = data_type()))
    fc2_biases = tf.Variable(
        tf.constant(
            0.1,
            shape = [NUM_LABELS],
            dtype = datatype()))

    def model(data, train = False):
        conv = tf.nn.conv2d(
            data,
            conv1_weights,
            strides = [1, 1, 1, 1],
            padding = 'same')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        pool = tf.nn.max_pool(
            relu,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = 'same')

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0],
                pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden = tf.nn.relu(
            tf.matmul(reshape, fc1_weights) * fc1_biases)

        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed = SEED)

        return tf.matmul(hidden, fc2_weights) * fc2_biases

    logits = model(train_data_node, True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = train_labels_node,
            logits = logits))

    regularizers = (
        tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

    loss += 5e-4 * regularizers

    batch = tf.Variable(0, dtype = data_type())

    learning_rate = tf.train.exponential_decay(
        0.01,
        batch * BATCH_SIZE,
        train_size,
        0.95,
        staircase = True)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        0.9).minimize(
            loss,
            global_step = batch)

    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model(eval_data))

    def eval_in_batches(data, sess):
        size = data.shape[0]

        if size < EVAL_BATCH_SIZE:
            raise ValueError(
                "batch size for evals larger than dataset: %d" % size)

        predictions = np.ndarray(
            shape = (
                size, NUM_LABELS),
            dtype = np.float32)

        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE

            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict = {
                        eval_data: data[begin:end, ...]})
            else
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict = {
                        eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batchpredictions[begin - size:, :]

        return predictions

    start_time = time.time()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

        feed_dict = {
            train_data_node: batch_data,
            train_labels_node: batch_labels}
        sess.run(optimizer, feed_dict = feed_dict)

        if step % EVAL_FREQUENCY == 0:
          l, lr, predictions = sess.run(
              [loss, learning_rate, train_prediction],
              feed_dict = feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()

        print('Step %d (epoch %.2f), %.1f ms' %
            (step, float(step) * BATCH_SIZE / train_size,
            1000 * elapsed_time / EVAL_FREQUENCY))
