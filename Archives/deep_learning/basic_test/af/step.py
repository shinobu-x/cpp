import numpy as np
import matplotlib.pylab as pl

def step(x):
  y = x > 0
  return y.astype(np.int)

if __name__ == '__main__':
  x = np.arange(-5.0, 5.0, 0.1)
  y = step(x)
  pl.plot(x, y)
  pl.ylim(-0.1, 1.1)
  pl.show()
