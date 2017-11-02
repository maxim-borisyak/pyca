import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.uint8_t STATE_t
ctypedef cnp.uint8_t[:, :] RULES_t

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef STATE_t[:, :, :] ca(RULES_t rules, STATE_t[:, :, :] initial, int steps):
  cdef STATE_t[:, :, :] buffer1 = initial
  cdef STATE_t[:, :, :] buffer2 = np.zeros_like(initial)

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
cdef STATE_t[:, :, :] ca_step(RULES_t rules, STATE_t[:, :, :] buffer, STATE_t[:, :, :] output):
  """
  Performs one step of cellular automaton according to the `rules`.

  Each cell has 1 of two states: 0 (dead) or 1 (alive). 
  
  This procedure changes status of each cell to rules[current_state, number of neighbors].

  :param rules: 2d array of shape (2, 9); rules[current_state, i] 
  :param buffer: buffer.
  :return: None.
  """

  cdef int i, j, k
  cdef int p, q
  cdef int w = buffer.shape[1], h = buffer.shape[2]
  cdef int neighboors

  for k in range(buffer.shape[0]):
    for i in range(w):
      for j in range(h):
        neighboors = 0

        p = (i + w - 1) % w
        q = (j + h - 1) % h
        neighboors += buffer[k, p, q]

        p = i
        q = (j + h - 1) % h
        neighboors += buffer[k, p, q]

        p = (i + 1) % w
        q = (j + h - 1) % h
        neighboors += buffer[k, p, q]

        p = (i + w - 1) % w
        q = j
        neighboors += buffer[k, p, q]

        p = (i + 1) % w
        q = j
        neighboors += buffer[k, p, q]

        p = (i + w - 1) % w
        q = (j + 1) % h
        neighboors += buffer[k, p, q]

        p = i
        q = (j + 1) % h
        neighboors += buffer[k, p, q]

        p = (i + 1) % w
        q = (j + 1) % h
        neighboors += buffer[k, p, q]

        output[k, i, j] = rules[buffer[k, i, j], neighboors]

  return output