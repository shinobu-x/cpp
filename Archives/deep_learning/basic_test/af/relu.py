import numpy as np
import matplotlib.pylab as pl

def relu(x):
 return np.maximum(0, x)

if __name__ == '__main__':
  x = np.arange(-5.0, 5.0, 0.1)
  y = relu(x)
  pl.plot(x, y)
  pl.show()
