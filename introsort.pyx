""" ================ IntroSort implementation in C ================== """
from c_utils cimport frand32, frand
from heapsort cimport hsort_c
from sorting cimport partition_c, median3, _swap
from insertsort cimport insertsort

from c_utils cimport log2, read_numpy

from utils import print_func_name
import numpy as np

cdef void introsort(double* a, size_t n):
    cdef size_t maxdepth = log2(n)
    isort(a, n, maxdepth)

cdef void isort(double* a, size_t n, size_t maxdepth, size_t depth=0):

    if n <= 1:
        return
    if n == 2:
        if a[0] > a[1]:
            _swap(a, 0, 1)
        return

    if n < 16:
        insertsort(a, n)
        return

    if depth >= maxdepth:
        hsort_c(a, n)
        return

    cdef size_t p_idx, idx, delta

    # p_idx = frand32() % n
    p_idx = frand() % n
    # p_idx = rand() % n
    # p_idx = 0 # first
    # p_idx = n - 1 # last
    # p_idx = median3(a, n) # median of 3

    idx = partition_c(a, n, p_idx)
    delta = idx + 1
    isort(a, idx, maxdepth, depth + 1)
    isort(a + delta, n - delta, maxdepth, depth + 1)


def introsort_py(arr):
    cdef:
        double* a
        size_t  n
    a, n = read_numpy(arr)
    introsort(a, n)

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_introsort():
    
    cdef:
        size_t n = 100
        double* a
        size_t i, j
        double [:] a_mv
    # np.random.seed(5)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        a_mv = np.sort(arr)
        introsort(a, n)
        for j in range(n):
            assert a[j] == a_mv[j]
