import numpy as np

import sys
import os.path as osp

sys.path.append(osp.join(osp.split(__file__)[0], '..'))

print(sys.path)

from pyca import ca, probabilistic_ca

import time

if __name__ == '__main__':
  rules = np.array(
    [
      [0, 0, 1.0e-2, 1, 1.0e-2, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0]
    ],
    dtype='float32'
  )

  print('Rules')
  print(rules)

  batch_size = 128
  n_steps = 128

  buffer = (np.random.randint(100, size=(batch_size, 128, 128), dtype='uint8') < 20).astype('uint8')

  start = time.time()
  probabilistic_ca(rules, buffer, n_steps)
  end = time.time()

  print('Time per step per sample: %.1e sec' % ((end - start) / batch_size / n_steps))
