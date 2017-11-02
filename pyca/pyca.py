import pyximport; pyximport.install()
from .pyca_ops import ca

import numpy as np

__all__ = [
  'ca',
  'game_of_life_rules'
]

game_of_life_rules = np.array(
    [
      [0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0]
    ],
    dtype='uint8'
  )