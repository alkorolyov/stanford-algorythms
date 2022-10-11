# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

from libc.stdlib cimport malloc, realloc, free, EXIT_FAILURE
from utils import print_func_name
import numpy as np


""" ################## Arrays in C ######################### """

ctypedef struct array_c:
    size_t maxsize
    size_t* items

cdef array_c* create_arr(size_t maxsize):
    cdef array_c* arr = <array_c*> malloc(sizeof(array_c))
    arr.maxsize = maxsize
    arr.items = <size_t*>malloc(sizeof(size_t) * maxsize)
    return arr

cdef void resize_arr(array_c* arr):
    arr.maxsize = 3 * arr.maxsize // 2
    arr.items = <size_t*>realloc(arr.items, arr.maxsize)

cdef void free_arr(array_c* arr):
    free(arr.items)
    free(arr)


""" ############################# UNIT TESTS ######################### """

def test_create_arr():
    print_func_name()
    cdef array_c* arr = create_arr(10)
    assert arr.maxsize == 10
    arr.items[9] = 1
    assert arr.items[9] == 1
    free_arr(arr)

def test_resize_arr():
    print_func_name()
    cdef array_c* arr = create_arr(10)
    resize_arr(arr)
    assert arr.maxsize == 15
    arr.items[14] = 1
    free_arr(arr)





