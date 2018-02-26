from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import gzip
import numpy as np

from six.moves import urllib

import tensorflow as tf

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
