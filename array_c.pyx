# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

from libc.stdlib cimport malloc, realloc, free
from utils import print_func_name, set_stdout, restore_stdout

from numpy cimport npy_intp, PyArray_SimpleNew, PyArray_DATA
cimport numpy as cnp
cnp.import_array()
import numpy as np


""" ################## Arrays in C ######################### """

cdef size_t max_arr(array_c * arr):
    cdef:
        size_t i
        size_t max_val = 0

    for i in range(arr.size):
        if arr.items[i] > max_val:
            max_val = arr.items[i]
    return max_val

cdef bint isin_arr(array_c* arr, size_t val):
    cdef size_t i
    for i in range(arr.size):
        if arr.items[i] == val:
            return True
    return False


cdef void reverse_arr(array_c * arr):
    cdef:
        size_t i
        size_t n = arr.size - 1

    for i in range(arr.size // 2):
        _swap(arr, i, n - i)

cdef array_c* list2arr(object py_obj):
    """
    Convert python object of integers and return C array
    :param py_obj: indexed Python object, ex: list, tuple, numpy 1D array
    :return: pointer to C array
    """
    cdef:
        i = 0
        n = len(py_obj)
        array_c* arr = create_arr(n)
    arr.size = n
    for i in range(n):
        arr.items[i] = py_obj[i]
    return arr


cdef object arr2numpy(array_c* arr):
    cdef:
        size_t i
        size_t* data

    np_arr = PyArray_SimpleNew(1, <npy_intp*>&arr.size, cnp.NPY_UINT64)
    data = <size_t*>PyArray_DATA(np_arr)
    for i in range(arr.size):
        data[i] = arr.items[i]
    return np_arr

cdef array_c* create_arr(size_t n):
    cdef:
        array_c* arr = <array_c*> malloc(sizeof(array_c))
        size_t i, val
    arr.capacity = n
    arr.size = 0
    arr.items = <size_t*>malloc(sizeof(size_t) * n)
    return arr

cdef array_c* create_arr_val(size_t n, size_t val):
    cdef:
        array_c* arr = create_arr(n)
        size_t i
    arr.size = n
    for i in range(n):
        arr.items[i] = val
    return arr

cdef void push_back_arr(array_c* arr, size_t val):
    cdef:
        size_t i = arr.size

    if arr.size == arr.capacity:
        resize_arr(arr)

    arr.items[i] = val
    arr.size += 1


cdef inline void resize_arr(array_c* arr):
    arr.capacity = 2 * arr.capacity
    arr.items = <size_t*>realloc(arr.items, arr.capacity * sizeof(size_t))

cdef void free_arr(array_c* arr):
    free(arr.items)
    free(arr)

cdef void print_array(array_c* arr):
    cdef size_t i

    if arr.size == 0:
        print("[]")
        return

    print("[", end="")
    for i in range(arr.size - 1):
        print(arr.items[i], end=", ")
    print(arr.items[arr.size - 1], end="]\n")


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """



def test_create_arr():
    print_func_name()
    cdef array_c* arr = create_arr(10)
    assert arr.capacity == 10
    arr.items[9] = 1
    assert arr.items[9] == 1
    free_arr(arr)

def test_resize_arr():
    print_func_name()
    cdef array_c* arr = create_arr(10)
    resize_arr(arr)
    assert arr.capacity == 20
    arr.items[19] = 1
    free_arr(arr)

def test_list2arr():
    print_func_name()
    l = [1, 2, 3]
    cdef array_c* arr = list2arr(l)
    assert l[0] == arr.items[0]
    assert l[1] == arr.items[1]
    assert l[2] == arr.items[2]
    assert arr.size == len(l)

def test_arr2numpy():
    print_func_name()
    cdef array_c* arr = create_arr(3)
    push_back_arr(arr, 1)
    push_back_arr(arr, 2)
    push_back_arr(arr, 3)
    np_arr = arr2numpy(arr)
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.size == 3
    assert np_arr[0] == 1
    assert np_arr[1] == 2
    assert np_arr[2] == 3
    free_arr(arr)

def test_swap():
    print_func_name()
    cdef array_c * arr = create_arr(3)
    arr.items[0] = 3
    arr.items[1] = 2
    arr.items[2] = 1
    arr.size = 3

    _swap(arr, 0, 2)

    assert arr.items[0] == 1
    assert arr.items[2] == 3
    free_arr(arr)


def test_reverse_even():
    cdef array_c * arr = create_arr(4)
    arr.items[0] = 3
    arr.items[1] = 2
    arr.items[2] = 1
    arr.items[3] = 0
    arr.size = 4

    reverse_arr(arr)

    assert arr.items[0] == 0
    assert arr.items[1] == 1
    assert arr.items[2] == 2
    assert arr.items[3] == 3
    free_arr(arr)

def test_reverse_odd():
    cdef array_c * arr = create_arr(3)
    arr.items[0] = 3
    arr.items[1] = 2
    arr.items[2] = 1
    arr.size = 3

    reverse_arr(arr)

    assert arr.items[0] == 1
    assert arr.items[1] == 2
    assert arr.items[2] == 3
    free_arr(arr)



def test_print():
    print_func_name()
    cdef array_c * arr = create_arr(10)
    arr.items[0] = 3
    arr.items[1] = 2
    arr.items[2] = 1
    arr.size = 3

    s = set_stdout()
    print_array(arr)
    out = s.getvalue()
    restore_stdout()

    assert out == '[3, 2, 1]\n'

    free_arr(arr)

def test_print_zero_length():
    print_func_name()
    cdef array_c * arr = create_arr(10)

    s = set_stdout()
    print_array(arr)
    out = s.getvalue()
    restore_stdout()

    assert out == '[]\n'
    free_arr(arr)
