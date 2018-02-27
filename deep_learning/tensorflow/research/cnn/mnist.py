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
def cnn_model_fn(features, labels, mode):
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
    activation = tf.nn.relu)

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
  pool2_flatten = tf.reshape(
    pool2,
    [-1, 7 * 7 * 64])

  # Dense layer
  # Densely connected layer with 1024 neurons
  # Input tensor shape: [batch_size, 7 * 7 * 64]
  # Output tensor shape: [batch_size, 1024]
  dense = tf.layers.dense(
    inputs = pool2_flatten,
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

  # Generate predictions
  # The logits layer of our model returns our predictions as raw values in a [
  # batch_size, 10] - demensioanl tensor. Let's convert these raw values into
  # two different formats that our model function can return.
  #
  # The predicted class for each example: A digit from 0 - 9.
  # The probabilities for each possible target class for each example: The pro-
  # bability that the example is a 0, is a 1, is a 2, etc.
  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(
      input = logits,
      axis = 1),
    # Adds softmax_layer to the graph. It is used for PREDICT and by the loggi-
    # ng_hook
    "probabilities": tf.nn.softmax(
      logits,
      name = "softmax_tensor")
  }
  # Compiles our prediction in a dict, and returns an EstimatorSpec object.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode = mode,
      predictions = prediction)

  # Calculates loss
  # For both training and evaluation, we need to define a loss function that m-
  # easures how closely the model's predictions match the target classes. For
  # multiclass classification problems like MNIST, cross entropy is typically
  # used as the loss metric.
  loss = tf.losses.sparse_softmax_cross_entropy(
    labels = labels,
    logits = logits)

  # Configures the Training Op
  # In the previous section, we defined loss for our CNN as the softmax cross
  # entrooy of the logts layer and our labels. Let's configure our model to op-
  # timize this loss value during training. We will use a learning rate of 0.00
  # 1 and stochastic gradient descent as the optimization algorithm.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate = 0.001)
    train_op = optimizer.minimize(
      loss = loss,
      global_step = tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
      mode = mode,
      loss = loss,
      train_op = train_op)

  # Adds evalution metrics
  eval_metrics_ops = {
    "accuracy": tf.metrics.accuracy(
      labels = labels,
      predictions = predictions["classes"])}
  return tf.estimator.EstimatorSpec(
    mode = mode,
    loss = loss,
    eval_metrics_ops = eval_metrics_ops)

def main(unused_argv):
  # Loads training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(
    mnist.train.labels,
    dtype = np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(
    mnist.test.labels,
    dtype = np.int32)

  # Creates the estimator
  # The model_fn argument specifies the model function to use for training, ev-
  # aluation. We pass the cnn_mode_fn. The model_dir argument specifies the di-
  # rectory where model data (checkpoints) will be saved.
  mnist_classifier = tf.estimator.Estimator(
    model_fn = cnn_model_fn,
    model_dir = "/tmp/mnist_convert_mode")

  # Sets up a logging hook
  # Since CNNs can take a while to train, let's set up some logging so we can
  # track progress during training. We can use tf.train.SessionRunHook to crea-
  # te a tf.train.Logging.TensorHook that will log the probability values from
  # the softmax layer of our CNN.
  #
  # We create the LoggingTensorHook, passing tensors_to_log to the tensors arg-
  # ument. We set every_n_iter = 50, which specifies that probabilities should
  # be logged every 50 steps of training.
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log,
    every_n_iter = 50)

  # Trains the model by creating train_input_fn and calling train() on mnist_c-
  # lassifier.
  #
  # In the numpy_input_fn call, we pass the training feature data and labels to
  # x (as a dict) and y, respectively. We set a batch_size of 100 (which means
  # that the model will train on minibatches of 100 examples at each step).
  # num_epochs = None means that the model will train until the specified numb-
  # er of steps is reached. We also set shuffle = True to shuffle the training
  # data. In the train call, we set steps = 20000 (which means the model will
  # train for 20000 steps total). We pass our logging_hook to the hooks argume-
  # nt, so that it will be triggered during training.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": train_data},
    y = train_labels,
    batch_size = 100,
    num_epochs = None,
    shuffle = True)
  mnist_classifier.train(
    input_fn = train_input_fn,
    steps = 20000,
    hooks = [logging_hook])

  # Evaluates the model
  # Once training is complete, we want to evaluate our model to determine its
  # accuracy on the MNIST test set. We call the evaluate method, which evaluat-
  # es the metrics we specified in eval_metric_ops argument in the model_fn.
  #
  # To create eval_input_fn, we set num_epochs = 1, so that the model evaluates
  # the metrics over one epoch of data and returns the result. We also set shu-
  # ffle = False to iterate through the data sequentially.
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": eval_data},
    y = eval_labels,
    num_epochs = 1,
    shuffle = False)
  eval_results = mnist_classifier.evaluate(
    input_fn = eval_input_fn)
  print(eval_result)

if __name__ == "__main__":
  tf.app.run()
