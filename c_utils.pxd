cimport numpy as np

cdef inline size_t fastrand(size_t seed=0):
  cdef size_t g_seed = (214013 * seed + 2531011)
  return (g_seed>>16)&0x7FFF

cdef:
    void err_exit(char* err_msg)
    (double *, size_t) read_numpy(np.ndarray[double] arr)