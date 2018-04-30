import numpy as np
import matplotlib.pylab as pl

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
  x = np.arange(-5.0, 5.0, 0.1)
  y = sigmoid(x)
  pl.plot(x, y)
  pl.ylim(-0.1, 1.1)
  pl.show()
