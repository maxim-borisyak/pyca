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
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE[:, :, :] ca(RULES rules, STATE[:, :, :] initial, int steps):
  cdef STATE[:, :, :] buffer1 = initial
  cdef STATE[:, :, :] buffer2 = np.zeros_like(initial)

  cdef int i

  for i in range(steps):
    ca_step(rules, buffer1, buffer2)
    buffer1, buffer2 = buffer2, buffer1

  initial[:] = buffer1[:]

  return initial

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cdef STATE[:, :, :] probabilistic_ca_step(PROB_RULES rules, STATE[:, :, :] buffer, STATE[:, :, :] output):
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
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE[:, :, :] probabilistic_ca(PROB_RULES rules, STATE[:, :, :] initial, int steps):
  cdef STATE[:, :, :] buffer1 = initial
  cdef STATE[:, :, :] buffer2 = np.zeros_like(initial)

  cdef int i

  for i in range(steps):
    probabilistic_ca_step(rules, buffer1, buffer2)
    buffer1, buffer2 = buffer2, buffer1

  initial[:] = buffer1[:]

  return initial
