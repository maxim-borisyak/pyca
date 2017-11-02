import sys
import os.path as osp
sys.path.append(osp.join(osp.split(__file__)[0], '..'))

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')

from pyca import ca, probabilistic_ca

import time

if __name__ == '__main__':
  rules = np.array(
    [
      [0.001, 0.05, 0.1, 0.8, 0.05, 0.01, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0]
    ],
    dtype='float32'
  )

  print('Rules:')
  print(rules)

  buffer = (np.random.randint(100, size=(2, 128, 128), dtype='uint8') < 35).astype('uint8')

  plt.ion()
  fig = plt.figure()

  for _ in range(1024):
    probabilistic_ca(rules, buffer, 1)

    plt.clf()
    plt.imshow(buffer[0])
    fig.canvas.draw()
    time.sleep(0.1)
