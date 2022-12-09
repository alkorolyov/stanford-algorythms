import random

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.stdio cimport printf
from quicksort cimport part_h
cimport numpy as np
np.import_array()

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

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_swap_c():
    
    cdef:
        double* a = [0.1, 0.2]
    swap(a, &a[1])
    assert a[0] == 0.2
    assert a[1] == 0.1


def test_frand():
    
    cdef:
        size_t i, rnd
    for i in range(128000):
        rnd = frand()
        assert rnd >= 0
        assert rnd < 1 << 16 #32767

def test_srand():
    
    sfrand(0)
    global seed
    assert seed == 0
    frand()
    assert seed != 0

def test_frand32():
    
    cdef:
        size_t i, rnd
        size_t n = 150
    for i in range(128000):
        rnd = frand32()
        assert rnd >= 0
        assert rnd < 1 << 32 # 2^32 - 1
        assert rnd % n < n
        assert rnd % n >= 0


def test_imed3():
    cdef:
        double* a = [0.1, 0.2, 0.3, 0.4, 0.5 , 0.6]
        double* b = [0.1, 0.2, 0.3, 0.4, 0.5 , 0.6]
        double* lo = a
        double* hi = &a[5]
        double* pp
        double p, p1, p2

    p = imed3(a, a + 5)
    assert p == med3(b, b + 5)


def test_med3_sse():
    cdef:
        double* a = [0.1, 0.2, 0.3, 0.4, 0.5 , 0.6]
        double* b = [0.1, 0.2, 0.3, 0.4, 0.5 , 0.6]
        double* lo = a
        double* hi = &a[5]
        double* pp
        double p, p1, p2

    p = med3(a, &a[5])
    pp = part_h(a, &a[5], p)

    p1, p2 = med3_sse(a, &a[5], pp)

    p = med3(b, &b[5])
    pp = part_h(b, &b[5], p)

    # print("scalar:", med3(b, pp), med3(&pp[1], &b[5]))
    # print("sse:", pp[1], pp[2])


    # assert p1 == med3(b, pp)
    # assert p2 == med3(&pp[1], &b[5])