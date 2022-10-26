#cython: language_level=3

from scipy.linalg import lu
import numpy as np
from time import time


def lu_time():
    cdef:
        size_t i
        size_t n = 100000

    res = np.empty((n, 3))
    A = np.random.rand(3, 3, n)

    cdef:
        double [:, ::1] r = res
        double [:, :] u

    start = time()
    for i in range(n):
        # A = np.random.rand(3, 3)
        _, _, U = lu(A[:, :, i])
        u = U
        r[i, 0] = u[0, 0]
        r[i, 1] = u[1, 1]
        r[i, 2] = u[2, 2]
        # print(res[i, 0])
        # print(A)
        # print(U)
    # print(res)
    print(res.mean(axis=0))
    print(f"{time() - start:.2f}s")