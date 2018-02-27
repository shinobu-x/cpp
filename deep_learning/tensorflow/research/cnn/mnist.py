# Convolutional neural networks (CNNs) are the current state-of-the-art model
# architecture of image classification task.
# CNNs apply a series of filters to the raw pixel data of an image to extract
# and learn higher-level features, which the model can then use for classifica-
# tion.
#
# CNNs contain three components:
# 1.Convolutional layers:
# Which apply a specified number of convolution filters to the image. For each
# subregion, the layer performs a set of mathematical operation to produce a s-
# ingle value in the output feature map. Convolutional layers then typically a-
# pply a ReLU activation function to the output to introduce nonlinearities in-
# to the model.
#
# 2.Pooling layers:
# Which downsample the image data extracted by the convolutional layers to red-
# uce the dimensionality of the feature map in order to decrease processing ti-
# me. A commonly used pooling algorithm is max pooling, which extracts subregi-
# on of the feature map (e.g., 2x2-pixel tiles), keeps their maximum value, and
# discards all other values.
#
# 3.Dense (Fully connected) layers:
# Which perform classification on the features extracted by the convolutional
# layers and downsampled by the pooling layers. In a dense layer, every node l-
# ayer, every node in the layer is connected to every node in the preceding la-
# yer.
#
# Typically, a CNN is composed of a stack of convolutional modules that perform
# feature extraction. Each module consists of a convolutional layer followed by
# a pooling layer. The last convolutional module is followed by one or more de-
# nse layers that perform classification. The final dense layer in a CNN conta-
# ins a single node for each target class in the model (all the possible class-
# es the model may predict), with a softmax activation function to generate a
# value between 0 - 1 for each node (the sum of all these softmax value is equ-
# al to 1). We can interpret the softmax values for a given image as relative
# measurements of how likely it is that the image falls into each target class.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Model function for CNN.
def cnn_model_fn(feature, labels, mode):
  # conv2d():
  # Constructs a two-dimensional convolutional layer. Takes number of filters,
  # filter kernel size, padding, and activation function as arguments.
  #
  # max_pooling2d():
  # Constructs a two-dimensional pooling layer using the max-pooling algorithm.
  # Takes pooling filter size and stride as arguments.
  #
  # dense():
  # Constructs a dense layer. Takes number of neurons and activation function
  # as arguments
  #
  # Each of these methods accepts a tensor as input and returns a transformed
  # tensor as output. This makes it easy to connect one layer to another, taki-
  # ng the output from one layer-creation method and apply it to as input to a-
  # nother.

  # Input layer
  # The methods in the layers module for creating convolutional and pooling la-
  # yers for two-dimensional image data expect input tensors to have a shape of
  # [batch_size, image_width, image_height, channels], defined as follows:
  #
  # batch_size: Size of the subset of examples to use when performing gradient
  # descent during training.
  # image_width: Width of the example images.
  # image_height: Height of the example eimages.
  # channels: Number of color channels in the example images. For color images,
  # the number of channels is 3(Red, Green, Blue). For monochrome images, there
  # is just 1 channel (Black).
  #
  # MNIST dataset is composed of monochrome 28x28 pixel images, so the desired
  # shape for our input layer is [batch_size, 28, 28, 1]
  # Reshapes x to 4D tensor: [batch_size, widht, height, channels].
  #
  # -1 for batch_size, which specifies that this dimension should be dynamical-
  # ly computed based on the number of input values in feature["x"], holding t-
  # he size of all other dimensions constant. This allows us to treat batch_si-
  # ze as hyperparameter that we can tune.
  # For example, if wee feed examples into our model in batches of 5, feature["
  # 5"] will contain 3920 values (one value for each pixel in each image), and
  # input_layer will have a shape of [5, 28, 28, 1].
  # Similarly, if we feed examples in batches of 100, feature["x"] will contai-
  # n 78400 values, and input_layer will have a shape of [100, 28, 28, 1].
  input_layer = tf.reshape(
    features["x"],
    [-1, 28, 28, 1])


  # Convolutional layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve widht and height.
  # Input tensor shape: [batch_size, 28, 28, 1].
  # Output tensor shape: [batch_size, 28, 28, 32].
  conv1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 32,
    kernel_size = [5, 5],
    padding = 'same',
    activation = tf.nn.relu)

  # Pooling layer #1
  # First max pooling layer with a 2x2 filter and stride of 2.
  # Input tensor shape: [batch_size, 28, 28, 32].
  # Output tensor shape: [batch_size, 14, 14, 32].
  pool1 = tf.layers.max_pooling2d(
    inputs = conv1,
    pool_size = [2, 2],
    strides = 2)

  # Convolutional layer #2
  # Compute 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input tensor shape: [batch_size, 14, 14, 32].
  # Output tensor shape: [batch_size, 14, 14, 64].
  conv2 = tf.layers.conv2d(
    inputs = pool1,
    filters = 64,
    kernel_size = [5, 5],
    padding = 'same',
    activation = tf.nn.relue)

  # Pooling layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2.
  # Input tensor shape: [batch_size, 14, 14, 64].
  # Output tensor shape: [batch_size, 7, 7, 64].
  pool2 = tf.layers.max_pooling2d(
    inputs = conv2,
    pool_size = [2, 2],
    strides = 2)

  # Flatten tensor into a batch of vectors.
  # Input tensor shape: [batch_size, 7, 7, 64].
  # Output tensor shape: [batch_size, 7 * 7 * 64].
  pool2_flatten = tr.reshape(
    pool2,
    [-1, 7 * 7 * 64])

  # Dense layer
  # Densely connected layer with 1024 neurons
  # Input tensor shape: [batch_size, 7 * 7 * 64]
  # Output tensor shape: [batch_size, 1024]
  dense = tf.layers.dense(
    input = pool2_flatten,
    units = 1024,
    activation = tf.nn.relu)

  # Adds dropout operation.
  # 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
    inputs = dense,
    rate = 0.4,
    training = mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # The final layer in our neural network is the logits layer, which will retu-
  # rn the raw values for our predictions. We create a dense layer with 10 neu-
  # ron (one for each target class 0 - 9), with linear activation(default)
  #
  # Input tensor shape: [batch_size, 1024]
  # Output tensor shape: [batch_size, 10]
  logits = tf.layers.dense(
    inputs = dropout,
    units = 10)


