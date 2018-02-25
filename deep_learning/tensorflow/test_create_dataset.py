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
      count = 0

      for _ in file_handles[-1]:
       count += 1
       if count == offset:
         break

  writers = []
  for i in range(FLAGS.output_shards):
    writers.append(
      tf.python_io.TFRecordWriter(
        "%s-%05i-of-%05i" % (output_file, i, output_shards)))

  reading_order = range(len(file_handlers)) * observations_per_class
  random.shuffle(reading_order)

  for c in reading_order:
    line = file_handles[c].readline()
    ink = None

    while ink is None:
      ink, class_name = parse_line(line)

      if ink is None:
        print("Couldnt't parse ink from '" + line + "'.")

    if class_name not in class_names:
      class_names.append(class_name)

    features = {}
    features["class_index"] = tf.train.Feature(
      int64_list = tf.train.Int64List(
        value = [class_names.index(class_name)]))

    features["ink"] = tf.train.Feature(
      float_list = tf.train.FloatList(
        value = ink.flatten()))

    features["shape"] = tf.train.Feature(
      int64_list = tf.train.Int64List(
        value = ink.shape))

    f = tf.train.Features(feature = features)
    example = tf.train.Example(features = f)
    writes[_pick_output_shared()].write(example.SerializeToString())

  # Closes all files.
  for w in writers:
    w.close()
  for f in file_handles:
    f.close()

  # Writes the class list.
  with tf.gfile.GFile(output_file + ".classes", "w") as f:
    for class_name in class_names:
      f.write(class_name + "\n")

  return class_names

def main(_):
  class_names = convert_data(
    FLAGS.ndjson_path,
    FLAGS.train_observations_per_class,
    os.path.join(
      FLAGS.output_path, "training.tfrecord"),
    class_names = [],
    output_shards = FLAGS.output_shards,
    offset = 0)

  convert_data(
    FLAGS.ndjson_path,
    FLAGS.eval_observations_per_class,
    os.path.join(FLAGS.output_path, "eval.tfrecord"),
    class_names = class_names,
    output_shards = FLAGS.output_shards,
    offset = FLAGS.train_observations_per_class)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument(
    "--ndjson_path",
    type = str,
    default = "",
    help = "Directory where the ndjson files are stored.")

  parser.add_argument(
    "--output_path",
    type = int,
    default = "",
    help = "Directory where to store the output TFRecord files.")

  parser.add_argument(
    "--train_observations_per_class",
    type = int,
    default = 10000,
    help = "How many items per class to load for training.")

  parser.add_argument(
    "--eval_observations_per_class",
    type = int,
    default = 1000,
    help = "How many times per class to load for evaluation.")

  parser.add_argument(
    "--output_shards",
    type = int,
    default = 10,
    help = "Number of shards for the output.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
