from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import gzip
import numpy as np

from six.moves import urllib

import tensorflow as tf

BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

def read(byte_stream):
  # Reads 4 bytes from byte stream as an unsigned 32bit integer and big endian.
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(byte_stream.read(4), dtype = dt)[0]

def check_image_file_header(file_name):
  # Validates that file name corresponds to images for the MNIST dataset.
  with tf.gfile.Open(file_name, 'rb') as f:
    magic = read(f)
    num_images = read(f)
    rows = read(f)
    cols = read(f)

    if magic != 2051:
      raise ValueError(
        'Invalid magic number %d in MNIST file %s' %
        (magic, f.name))

    if rows != 28 or cols != 28:
      raise ValueError(
        'Invalid MNIST file %s: Unxpected images found %dx%d' %
        (f.name, rows, cols))

def check_labels_file_header(file_name):
  # Validates that file_name corresponds to labels for the MNIST dataset
  with tf.gfile.Open(file_name, 'rb') as f:
    magic = read(f)
    num_items = read(f)
    if magic != 2049:
      raise ValueError(
        'Invalid magic number %d in MNIST file %s' %
        (magic, f.name))

def maybe_download(directory, file_name):
  # Downloads (and unzip) a file from the MNIST dataset if not already there
  file_path = os.path.join(directory, filename)
  if tf.gfile.Exists(file_path):
    return file_path
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)

  # Mirror of http://yann.lecun.com/exdb/mnist/
  url = BASE_URL + filename + '.gz'

  zipped_file_path = filepath + '.gz'
  print(
    'Downloading %s to %s' %
    (url, zipped_file_path))
  urllib.request.urlretrieve(url, zipped_file_path)
  with gzip.open(
    zipped_file_path, 'rb') as in_file, open(
    file_path, 'wb') as out_file:
    shutil.copyfileobj(in_file, out_file)
  os.remove(zipped_file_path)
  return file_path

def dataset(direcotry, image_files, label_files):
  image_files = maybe_download(directory, image_files)
  label_files = maybe_download(directory, label_files)

  check_image_file_header(image_files)
  check_label_file_header(label_files)

  def decode_image(image):
    # Normalizes from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf. uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0

  def decode_label(label):
    # tf.string -> [tf.uint8]
    label = tf.decode_raw(label, tf.uint8)
    # Label is a scalar
    label = tf.reshape(label, [])
    return tf.to_int32(label)

  images = tf.data.FixedLengthRecordDataset(
    image_files,
    28 * 28,
    header_bytes = 16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
    label_files,
    1,
    header_bytes = 8).map(decode_label)
  return tf.data.Dataset.zip((images, labels))

def train(directory):
  # tf.data.Dataset object for MNIST training data.
  return dataset(directory,
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte')

def test(directory):
  # tf.data.Dataset object for MNIST test data.
  return dataset(directory,
    't10k-images-idx3-ubyte',
    't10k-lables-idx1-ubyte')
