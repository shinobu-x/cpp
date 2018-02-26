import numpy as np

def softmax1(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def softmax2(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)
