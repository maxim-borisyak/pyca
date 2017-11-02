import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')

from pyca import ca

import time

if __name__ == '__main__':
  rules = np.array(
    [
      [0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0]
    ],
    dtype='uint8'
  )

  buffer = (np.random.randint(100, size=(2, 128, 128), dtype='uint8') < 20).astype('uint8')

  start = time.time()
  ca(rules, buffer, 1024)
  end = time.time()

  print('Time per step per sample: %.1e sec' % ((end - start) / 1024 / 2))

  buffer = (np.random.randint(100, size=(2, 32, 32), dtype='uint8') < 35).astype('uint8')
  ca(rules, buffer, 16)

  fig = plt.figure()
  plt.imshow(buffer[0])
  plt.show()