# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

from libc.stdlib cimport malloc, realloc, free, EXIT_FAILURE
from utils import print_func_name, set_stdout, restore_stdout
import numpy as np



""" ################## Arrays in C ######################### """

ctypedef struct array_c:
    size_t maxsize
    size_t* items

cdef array_c* list2arr(list py_list):
    """
    Read python list of integers and return C array
    :param a: list Python object
    :return: pointer to C array
    """
    cdef:
        i = 0
        n = len(py_list)
        array_c* arr = create_arr(n)
    for i in range(n):
        arr.items[i] = py_list[i]
    return arr


cdef array_c* create_arr(size_t maxsize):
    cdef array_c* arr = <array_c*> malloc(sizeof(array_c))
    arr.maxsize = maxsize
    arr.items = <size_t*>malloc(sizeof(size_t) * maxsize)
    return arr

cdef void resize_arr(array_c* arr):
    arr.maxsize = 2 * arr.maxsize
    arr.items = <size_t*>realloc(arr.items, arr.maxsize * sizeof(size_t))

cdef void free_arr(array_c* arr):
    free(arr.items)
    free(arr)

cdef void print_array(array_c* arr, size_t n):
    cdef size_t i

    if n > arr.maxsize:
        print("array print error: print size too long")
        exit(EXIT_FAILURE)

    if n == 0:
        print("[]")
        return

    print("[", end="")
    for i in range(n - 1):
        print(arr.items[i], end=", ")
    print(arr.items[n - 1], end="]\n")


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_list2arr():
    print_func_name()
    l = [1, 2, 3]
    cdef array_c* arr = list2arr(l)
    assert l[0] == arr.items[0]
    assert l[1] == arr.items[1]
    assert l[2] == arr.items[2]

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
    assert arr.maxsize == 20
    arr.items[19] = 1
    free_arr(arr)

def test_print():
    print_func_name()
    cdef array_c * arr = create_arr(10)
    arr.items[0] = 3
    arr.items[1] = 2
    arr.items[2] = 1

    s = set_stdout()
    print_array(arr, 3)
    out = s.getvalue()
    restore_stdout()

    assert out == '[3, 2, 1]\n'

    free_arr(arr)

def test_print_zero_length():
    print_func_name()
    cdef array_c * arr = create_arr(10)

    s = set_stdout()
    print_array(arr, 0)
    out = s.getvalue()
    restore_stdout()

    assert out == '[]\n'
    free_arr(arr)
