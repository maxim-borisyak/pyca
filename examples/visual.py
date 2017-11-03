import sys
import os.path as osp
sys.path.append(osp.join(osp.split(__file__)[0], '..'))

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')

from pyca import ca, prob_ca

import time

if __name__ == '__main__':
  rules = np.array(
    [
      [0.0, 0.0, 0.0, 1.0, 0, 0.0, 0, 0, 0],
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
    buffer = prob_ca(rules, buffer, 16)

    plt.clf()
    plt.imshow(buffer[0])
    fig.canvas.draw()
    time.sleep(0.1)
