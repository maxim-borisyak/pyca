import numpy as np

import sys
import os.path as osp

sys.path.append(osp.join(osp.split(__file__)[0], '..'))

from pyca import ca, prob_ca, uniform_prob_ca_stream

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

  batch_size = 32
  n_steps = 16
  n = 128

  stream = uniform_prob_ca_stream(rules, n_steps, shape=(batch_size, 64, 64), init_prob=0.2)

  start = time.time()
  for _ in range(n):
      next(stream)
  end = time.time()

  print('Time per step per sample: %.1e sec' % ((end - start) / batch_size / n_steps / n))
  print('Time per batch: %.1e sec' % ((end - start) / n))
