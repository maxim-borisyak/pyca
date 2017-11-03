try:
  from .pyca_ops import ca, uniform_ca, uniform_ca_stream
  from .pyca_ops import prob_ca, uniform_prob_ca, uniform_prob_ca_stream
except ImportError:
  import pyximport; pyximport.install()
  from .pyca_ops import ca, uniform_ca, uniform_ca_stream
  from .pyca_ops import prob_ca, uniform_prob_ca, uniform_prob_ca_stream

import numpy as np

__all__ = [
  'ca', 'uniform_ca', 'uniform_ca_stream',
  'prob_ca', 'uniform_prob_ca', 'uniform_prob_ca_stream',
  'game_of_life_rules'
]

game_of_life_rules = np.array(
    [
      [0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0]
    ],
    dtype='uint8'
  )
