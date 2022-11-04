from c_utils cimport read_numpy, print_arr

import numpy as np
from utils import print_func_name

def insertsort_py(arr: np.array):
    cdef:
        double* a
        size_t  n
    a, n = read_numpy(arr)
    insertsort(a, n)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_insert():
    print_func_name()
    cdef double* a = [0.1, 0.2, 0.3]
    insert(a, 3)
    # print_arr(a, 3)
    assert a[0] == 0.1
    assert a[1] == 0.2
    assert a[2] == 0.3

    cdef double* b = [0.1, 0.2, 0.15]
    # insert(b, 2, b[2])
    insert(b, 3)
    # print_arr(b, 3)
    assert b[0] == 0.1
    assert b[1] == 0.15
    assert b[2] == 0.2

    cdef double* c = [0.1, 0.2, 0.1]
    # insert(c, 2, c[2])
    insert(c, 3)
    # print_arr(c, 3)
    assert c[0] == 0.1
    assert c[1] == 0.1
    assert c[2] == 0.2

    cdef double* d = [0.1, 0.2, 0.0]
    # insert(d, 2, d[2])
    insert(d, 3)
    # print_arr(d, 3)
    assert d[0] == 0.0
    assert d[1] == 0.1
    assert d[2]  == 0.2

def test_insertsort():
    print_func_name()
    cdef:
        size_t n = 16
        double* a
        size_t i, j
        double [:] a_mv
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        a_mv = np.sort(arr)
        insertsort(a, n)
        for j in range(n):
            assert a[j] == a_mv[j]






