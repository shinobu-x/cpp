from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

"""
#1 Convolutional Layer1: Applies 32 5x5 filters (extracting 5x5-pixel subegions
), with ReLU activation function
#2 Pooling Layer1: Performs max pooling with a 2x2 filter and stride of 2 (whi-
ch specifies that pooled regions do not overlap)
#3 Convolutional Layer2: Applies 64 5x5 filters, with ReLU activation function
#4 Pooling Layer2: Performs max pooling with a 2x2 filter and stride of 2
#5 Dense Layer1: 1024 neurons, with dropout regularization rate of 0.4 (probab-
ility of 0.4 that any given element will be dropped during training)
#6 Dense Layer2 (Logits Layer): 10 neurons, one for each digit target class (0-
9)
"""
"""
Methods in the tf.layers module:
#1 conv2d():
 Conctructs a two-dimensional convolutional layer, take number of filters, fil-
 ter kernel size, padding, and activation function as arguments
#2 max_pooling2d():
 Constructs a two-dimensional pooling layer using the max-pooling algorithm, t-
 ake pooling filter size and stride as arguments
#3 dense():
 Constructs a dense layer, take number of neurons and activation function as a-
 rguments
"""
def cnn_model_fn(features, labels, mode):
    # Reshapes x to 4D tensor
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # Computes 32 features using a 5x5 filter with ReLU activation
    # Padding is added to preserve width and height
    # Input Tensor Shape:  [batch_size, 28, 28,  1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2, 2],
        strides = 2)

    # Computes 64 features using a 5x5 filter
    # Padding is added to preserve width and height
    # Input Tensor Shape:  [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 62]
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch-size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2, 2],
        strides = 2)

    # Input Tensor Shape:  [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(
        pool2,
        [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons
    # Input Tensor Shape:  [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs = pool2_flat,
        units = 1024,
        activation = tf.nn.relu)

    # Add dropout operation
    # 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN)

    # Input Tensor Shape:  [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(
        inputs = dropout,
        units = 10)

    predictions = {
        "classes": tf.argmax(
            input = logits,
            axis = 1),
        "probabilities": tf.nn.softmax(
            logits,
            name = "softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions)

    # Calculates loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels = labels,
        logits = logits)

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode = mode,
            loss = loss,
            train_op = train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels = labels,
            predictions = predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        eval_metric_ops = eval_metric_ops)

def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(
        mnist.train.labels,
        dtype = np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(
        mnist.test.labels,
        dtype = np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn,
        model_dir = "tmp")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log,
        every_n_iter = 50)

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

    eval_input_fn = tf.esimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False)

    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
