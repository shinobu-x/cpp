from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib

import tensorflow as tf

FLAGS = None

# pylint: disable = line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable = line-too-long

class NodeLookup(object):
    def __init__(
        self,
        label_lookup_path = None,
        uid_lookup_path = None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir,
                'imagenet_2012_challenge_label_map_proto.pbtxt')

        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir,
                'imagenet_synset_to_human_label_map.txt')

        self.node_lookup = self.load(
            label_lookup_path,
            uid_lookup_path)

    def load(
        self,
        label_lookup_path,
        uid_lookup_path):
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal(
                'File does not exist %s',
                label_lookup_path)

        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal(
                'File does not exist %s',
                uid_lookup_path)

        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')

        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()

        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                mode_id_to_uid[target_class] = target_class_string[1: -2]

        node_id_to_name = []

        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal(
                    'Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup
            return ''
        return self.node_lookup[node_id]
