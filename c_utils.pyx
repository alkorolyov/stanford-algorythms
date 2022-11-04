import random

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.stdio cimport printf
cimport numpy as np

from utils import print_func_name
from time import time

# initialize random seeds
global seed
seed = <size_t>time()

global rng_state
global rng_inc
rng_state = <size_t>time()
rng_inc = <size_t>time()

cdef void err_exit(char* err_msg):
    printf(err_msg)
    exit(1)

cdef (double*, size_t) read_numpy(np.ndarray[double] arr):
    cdef:
        np.npy_intp *dims
        double *data
    if arr.flags['C_CONTIGUOUS']:
        dims = np.PyArray_DIMS(arr)
        data = <double*>np.PyArray_DATA(arr)
        return data, <size_t>dims[0]
    else:
        print('Array is non C-contiguous')
        exit(1)

cdef void print_arr(double* arr, size_t n):
    cdef size_t i
    print("[", end="")
    for i in range(n - 1):
        print(arr[i], end=", ")
    print(arr[n - 1], end="]\n")

def rand_py():
    cdef size_t i, rnd
    for i in range(1000):
        rnd = rand()
    return rnd

def frand_py():
    cdef size_t i, rnd
    for i in range(1000):
        rnd = frand()
    return rnd

def frand32_py():
    cdef size_t i, rnd
    for i in range(1000):
        rnd = frand32()
    return rnd

def test_frand():
    print_func_name()
    cdef:
        size_t i, rnd
    for i in range(128000):
        rnd = frand()
        assert rnd >= 0
        assert rnd < 1 << 16 #32767

def test_srand():
    print_func_name()
    sfrand(0)
    global seed
    assert seed == 0
    frand()
    assert seed != 0

def test_frand32():
    print_func_name()
    cdef:
        size_t i, rnd
        size_t n = 150
    for i in range(128000):
        rnd = frand32()
        assert rnd >= 0
        assert rnd < 1 << 32 # 2^32 - 1
        assert rnd % n < n
        assert rnd % n >= 0

