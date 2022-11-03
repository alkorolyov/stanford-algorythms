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

def time_rand():
    DEF n = int(1e9)
    cdef:
        size_t i
        size_t seed = rand()
        pcg32_t rng
        size_t rnd = 0


    start = time()
    for i in range(n):
        rnd = frand()
    print(f"frand() {time() - start:.3f}s", end=" ")
    print(f"{hex(rnd)}")

    start = time()
    for i in range(n):
        rnd = frand32()
    print(f"frand32() {time() - start:.3f}s", end=" ")
    print(f"{hex(rnd)}")


    srand32(&rng, seed)
    start = time()
    for i in range(n):
        rnd = frand32_loc(&rng)
    print(f"frand32_loc() {time() - start:.3f}s", end=" ")
    print(f"{hex(rnd)}")


    srand(seed)
    start = time()
    for i in range(n // 10000):
        rnd = rand()
    print(f"rand() {(time() - start) * 10000:.3f}s", end=" ")
    print(f"{hex(rnd)}")


    # start = time()
    # for i in range(n // 1000):
    #     _rdrand64_step(&rnd)
    # print(f"_rdrand64_step() {(time() - start) * 1000:.3f}s", end=" ")
    # print(f"{hex(rnd)}")
    #
    # start = time()
    # for i in range(n // 1000):
    #     _rdrand32_step(&rnd)
    # print(f"_rdrand32_step() {(time() - start) * 1000:.3f}s", end=" ")
    # print(f"{hex(rnd)}")
    #
    # start = time()
    # for i in range(n // 1000):
    #     _rdrand16_step(&rnd)
    # print(f"_rdrand16_step() {(time() - start) * 1000:.3f}s", end=" ")
    # print(f"{hex(rnd)}")


