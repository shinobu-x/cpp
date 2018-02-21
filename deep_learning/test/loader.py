import os
import cPickle
import gzip
import numpy as np
from urllib import urlretrieve
from urlparse import urljoin

def download():
    base_url = 'http://deeplearning.net/data/mnist/'
    base_dir = '/tmp'
    fname = 'mnist.pkl.gz'
    target_fname = os.path.join(base_dir, fname)
    url = urljoin(base_url, fname)
    urlretrieve(url, target_fname)

def load():
    path = '/tmp/mnist.pkl.gz'
    f = gzip.open(path, 'rb')
    training, validation, testing = cPickle.load(f)
    f.close()
    return (training, validation, testing)

def loader():
    training, validation, testing = load()
    training_inputs = [np.reshape(x, (784, 1))
        for x in training[0]]
    training_results = [vectorization(y)
        for y in training[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1))
        for x in validation[0]]
    validation_data = zip(validation_inputs, validation[1])
    testing_inputs = [np.reshape(x, (784, 1))
        for x in testing[0]]
    testing_data = zip(testing_inputs, testing[1])
    return (training_data, validation_data, testing_data)

def vectorization(j):
    """
    Return a 10 demensional unit vector with a 1.0 in the jth position and
    zeros.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
