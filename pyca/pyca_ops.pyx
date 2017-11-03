import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.float32_t float32

ctypedef cnp.uint8_t STATE
ctypedef cnp.uint8_t[:, :] RULES
ctypedef float32[:, :] PROB_RULES

from libc.stdlib cimport rand, srand

cdef extern from "limits.h":
    int INT_MAX

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float32 frand() nogil:
  return (rand() % INT_MAX) / (<float32> INT_MAX)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cdef inline int neighbors(STATE[:, :, :] buffer, int i, int j, int k, int w, int h):
  cdef int n = 0
  cdef int p, q

  p = (i + w - 1) % w
  q = (j + h - 1) % h
  n += buffer[k, p, q]

  p = i
  q = (j + h - 1) % h
  n += buffer[k, p, q]

  p = (i + 1) % w
  q = (j + h - 1) % h
  n += buffer[k, p, q]

  p = (i + w - 1) % w
  q = j
  n += buffer[k, p, q]

  p = (i + 1) % w
  q = j
  n += buffer[k, p, q]

  p = (i + w - 1) % w
  q = (j + 1) % h
  n += buffer[k, p, q]

  p = i
  q = (j + 1) % h
  n += buffer[k, p, q]

  p = (i + 1) % w
  q = (j + 1) % h
  n += buffer[k, p, q]

  return n

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cdef STATE[:, :, :] ca_step(RULES rules, STATE[:, :, :] buffer, STATE[:, :, :] output):
  """
  Performs one step of cellular automaton according to the `rules`.

  Each cell has 1 of two states: 0 (dead) or 1 (alive).

  This procedure changes status of each cell to rules[current_state, number of neighbors].

  :param rules: 2d array of shape (2, 9); rules[current_state, i]
  :param buffer: buffer.
  :return: None.
  """

  cdef int i, j, k
  cdef int w = buffer.shape[1], h = buffer.shape[2]
  cdef int n

  for k in range(buffer.shape[0]):
    for i in range(w):
      for j in range(h):
        n = neighbors(buffer, i, j, k, w, h)
        output[k, i, j] = rules[buffer[k, i, j], n]

  return output

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE[:, :, :] ca(RULES rules, STATE[:, :, :] initial, int steps, STATE[:, :, :] buffer = None):
  cdef int i

  if buffer is None:
    buffer = np.zeros_like(initial)

  for i in range(steps):
    ca_step(rules, initial, buffer)
    initial, buffer = buffer, initial

  return initial

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE[:, :, :] uniform_ca(RULES rules, STATE[:, :, :] initial, int steps, float32 init_prob, STATE[:, :, :] buffer = None):
  cdef int i, j, k

  for i in range(initial.shape[0]):
    for j in range(initial.shape[1]):
      for k in range(initial.shape[2]):
        initial[i, j, k] = 1 if frand() < init_prob else 0

  return ca(rules, initial, steps, buffer)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
def uniform_ca_stream(RULES rules, int steps, shape, float32 init_prob):
  initial = np.ndarray(shape=shape, dtype='uint8')
  buffer = np.ndarray(shape=shape, dtype='uint8')

  while True:
    yield uniform_ca(rules, initial, steps, init_prob, buffer)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cdef STATE[:, :, :] prob_ca_step(PROB_RULES rules, STATE[:, :, :] buffer, STATE[:, :, :] output):
  """
  Performs one step of probabilistic cellular automaton according to the `rules`.

  Each cell has 1 of two states: 0 (dead) or 1 (alive).

  This procedure changes status of each cell to 1 with probability rules[current_state, number of neighbors].

  :param rules: 2d array of shape (2, 9); rules[current_state, i]
  :param buffer: buffer.
  :return: `output` array
  """

  cdef int i, j, k
  cdef int w = buffer.shape[1], h = buffer.shape[2]
  cdef int n
  cdef float32 r

  for k in range(buffer.shape[0]):
    for i in range(w):
      for j in range(h):
        n = neighbors(buffer, i, j, k, w, h)
        output[k, i, j] = 1 if rules[buffer[k, i, j], n] > frand() else 0

  return output

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE[:, :, :] prob_ca(PROB_RULES rules, STATE[:, :, :] initial, int steps, STATE[:, :, :] buffer = None):
  cdef int i

  if buffer is None:
    buffer = np.zeros_like(initial)

  for i in range(steps):
    prob_ca_step(rules, initial, buffer)
    initial, buffer = buffer, initial

  return initial

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE[:, :, :] uniform_prob_ca(PROB_RULES rules, STATE[:, :, :] initial, int steps, float32 init_prob, STATE[:, :, :] buffer = None):
  cdef int i, j, k

  for i in range(initial.shape[0]):
    for j in range(initial.shape[1]):
      for k in range(initial.shape[2]):
        initial[i, j, k] = 1 if frand() < init_prob else 0

  return prob_ca(rules, initial, steps, buffer)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
def uniform_prob_ca_stream(PROB_RULES rules, int steps, shape, float32 init_prob):
  initial = np.ndarray(shape=shape, dtype='uint8')
  buffer = np.ndarray(shape=shape, dtype='uint8')

  while True:
    yield uniform_prob_ca(rules, initial, steps, init_prob, buffer)
