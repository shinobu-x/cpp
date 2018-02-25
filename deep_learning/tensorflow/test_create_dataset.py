from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import sys

import numpy as np

import tensorflow as tf

# Parses an ndjson line and return ink (as np array) and class name.
def parse_line(ndjson_line):
  sample = json.loads(ndjson_line)
  class_name = sample["word"]

  if not class_name:
    print("Empty class_name")
    return None, None

  inkarray = sample["drawing"]
  stroke_lengths = [len(stroke[0]) for stroke in inkarray]
  rotal_points = sum(stroke_lengths)
  np_ink = np.zeros(
    (total_points, 3),
    dtype = np.float32)
  current_t = 0

  if not inkarray:
    print("Empty inkarray")
    return None, None

  for stroke in inkarray:
    if (len(stroke[0]) != len(stroke[1]):
      print("Inconsistent number of x and y coordinates.")
      return None, None

    for i in [0, 1]:
      np_ink[current_t:(
        current_t + len(stroke[0])), i] = stroke[i]

    current_t += len(stroke[0])
    np_ink[current_t - 1, 2] = 1 # stroke_end

  # Preprocessing.
  # 1.Size normalization.
  lower = np.min(np_ink[:, 0:2], axis = 0)
  upper = np.max(np_ink[:, 0:2], axis = 0)
  scale = upper - lower
  scale[scale == 0] = 1
  np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale

  # 2.Computes deltas.
  np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
  np_ink = np_ink[1:, :]

  return np_ink, class_name

# Converts training data from ndjson files into tf.Example in tf.Record.
#
# @training_data_dir: Path to the directory containining the training data set.
#  The training data is stored in that directory as ndjson files.
# @observations_per_class: The number of items to load per class.
# @output_file: Path to where to write the output.
# @class_names: Array with class names.
#  Which will be automatically created if not passed in.
# @output_shards: The number of shards to write the output in.
# @offset: The number of items to skip at the begining of each file.
#
# Returns:
#  class_names: The class names as string.
#   class_names[classes[i]] is the textual representation of the class of the i-
#   -th data point.
def convert_data(
  training_data_dir,
  observations_per_class,
  output_file,
  class_names,
  output_shards = 10,
  offset = 0):

  def _pick_output_shard():
    return random.randint(0, output_shareds - 1)

  file_handles = []
  # Opens all input files
  for file_name in sorted(tf.gfile.ListDirectory(training_data_dir)):
    if not file_name.endswith(".ndjson"):
      print(
        "Skipped",
        file_name)
      continue

    file_handles.append(
      tf.gfile.GFile(
        os.path.join(training_data_dir, file_name), "r"))
    # Fast forwards all files to skip the offset.
    if offset:
