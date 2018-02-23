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

        while tf.gfile.GFile(filepath) as f:
            size = f.size()

        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath

def extract_data(filename, num_images):
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
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
