from libc.stdlib cimport malloc, free, calloc, rand
from array_c cimport array_c, create_arr, free_arr, push_back_arr, isin_arr, count_arr, py2arr, print_array

import cython
from cython.parallel import prange
from time import time


cdef hashtable* make_hashtable(size_t n):
    cdef:
        hashtable * h = <hashtable*>malloc(sizeof(hashtable))
    h.items = <size_t*>calloc(n + 1, sizeof(size_t)) # zero-terminated membuffer
    h.capacity = n
    h.size = 0
    return h


cdef void free_hashtable(hashtable* h):
    free(h.items)
    free(h)


cdef void print_ht(hashtable* h):
    print("=== size:", h.size)
    for i in range(h.capacity):
        print(f"[{i}]: ", h.items[i])


cdef size_t twosum(hashtable* h, long long int t) nogil:
    """
    Test for integer pairs in hashtable, such that x + y = t    
    """
    cdef:
        size_t i, j
        size_t cnt = 0
        long long int x
    for i in range(h.capacity):
        x = h.items[i]
        if x:
            if search(h, t - x):
                return 1
    return 0


def test_twosum_c(verbose=True):
    if verbose:
        print()

    start = time()
    with open("2sum.txt", "r") as f:
        py_list = [int(s) for s in f.readlines()]

    cdef:
        long int i
        size_t cnt = 0
        long long int x, t
        hashtable* h = make_hashtable(2002823)

    if verbose:
        print(f"read_file: {time() - start:.2f}s")

    start = time()
    for i in range(len(py_list)):
        x = <long long int>py_list[i]
        if not search(h, x):
            insert(h, x)
    if verbose:
        print(f"make_hashtable: {time() - start:.2f}s")

    start = time()
    # for i in prange(20001, nogil=True):
    #     t = -10000 + i
    #     cnt += twosum(h, t)

    for i in range(64):
        t = -10000 + i
        cnt += twosum(h, t)

    if verbose:
        print(f"twosum: {time() - start:.2f}s")

    free_hashtable(h)
    return cnt


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

import numpy as np

def test_insert():
    cdef:
        size_t x = rand()
        hashtable* h = make_hashtable(7)
    insert(h, x)
    assert h.size == 1
    assert search(h, x)
    assert not search(h, x + 1)
    free_hashtable(h)

def test_search_rnd():
    cdef:
        size_t i
        # size_t n = 102100
        size_t n = 50000
        hashtable* h = make_hashtable(102197)
        long long int [:] arr = np.random.randint(-100000, 100000, n, dtype=np.int64)

    for i in range(n):
        insert(h, arr[i])

    for i in range(n):
        assert search(h, arr[i])

    free_hashtable(h)


def test_delete_single():
    cdef:
        size_t i
        size_t x = rand()
        hashtable* h = make_hashtable(7)
    insert(h, x)
    delete(h, x)
    assert not search(h, x)
    assert h.size == 0
    for i in range(h.capacity):
        assert h.items[i] == 0
    free_hashtable(h)

def test_delete_collision():
    cdef:
        size_t i
        hashtable* h = make_hashtable(5)

    insert(h, 1)
    insert(h, 6)
    insert(h, 2)
    insert(h, 1)
    # print()
    # print_ht(h)

    delete(h, 1)
    # print_ht(h)

    delete(h, 6)
    # print_ht(h)

    delete(h, 1)
    # print_ht(h)

    delete(h, 2)
    # print_ht(h)

    assert h.size == 0
    free_hashtable(h)


def test_delete_rnd():
    cdef:
        size_t i
        size_t n = 90000
        hashtable* h = make_hashtable(102197)
        long long int [:] arr = np.random.randint(-100000, 100000, n, dtype=np.int64)
    for i in range(n):
        insert(h, arr[i])
    # print()
    # print("h.size", h.size)
    for i in range(n):
        delete(h, arr[i])
    for i in range(n):
        assert not search(h, arr[i])
    for i in range(h.capacity):
        assert h.items[i] == 0
    # print("h.size", h.size)
    assert h.size == 0

def test_twosum():
    cdef:
        size_t i
        hashtable* h = make_hashtable(7)
    insert(h, 1)
    insert(h, -1)
    assert twosum(h, 0) == 1
    insert(h, -2)
    insert(h, 2)
    assert twosum(h, 0) == 1
    insert(h, 3)
    assert twosum(h, 4) == 1
    free_hashtable(h)