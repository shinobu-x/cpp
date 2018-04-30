import numpy as np

def sigmoid(x):
    """
    range: log((1 - 10^-15) / 10^-15)
    """
    range = 34.538776394910684
    if x <= -range:
      return 1e-15
    if x >= range:
      return 1.0 - 1e-15
    return 1.0 / (1.0 + np.exp(-x))
