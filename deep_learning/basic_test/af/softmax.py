import numpy as np
import matplotlib.pylab as pl

def softmax(x):
  e_x = np.exp(x)
  return e_x / np.sum(e_x)

if __name__ == '__main__':
  x = np.arange(-5, 5, 0.1)
  y = softmax(x)
  pl.plot(x, y)
  pl.show()
