import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.s = sizes
        self.biases = [np.random.randn(y, 1)
            for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
      for b, w in zip(self.biases, self.weights):
          a = sigmoid(np.dot(w, a) + b)
      return a

    def sgd(self, training, epochs, mini_batch_size, eta, test = None):
      if test : n = len(test)
      m = len(training)
      for j in xrange(epochs):
          random.shuffle(training)
          mini_batches = [
              training[k:k + mini_batch_size]
                  for k in xrange(0, n, mini_batch_size)]
          for mini_batch in mini_batches:
              self.update_mini_batch(mini_batch, eta)
          if test:
              print "Epoch {0}: {1} / {2}".format(
                  j, self.evaluate(test), n)
          else:
              print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape)
            for b in self.biases]
        nabla_w = [np.zeros(w.shape)
            for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropargation(x, y)
            nabla_b = [nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)]

    def backpropargation(self, x, y):
        nabla_b = [np.zeros(b.shape)
            for b in self.biases]
        nabla_w = [np.zeros(w.shape)
            for w in self.weights]
        # Feedforward
        activation = x
        # List to store all the activation, layer by layer
        activations = [x]
        # List to store all the Z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward
        delta = self.cost_derivation(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluation(self, test):
        results = [(np.argmax(self.feedforward(x)), y)
            for (x, y) in test]
        return sum(int(x == y)
            for (x, y) in results)

    def cost_derivation(self, activations, y):
      return (activations - y)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
