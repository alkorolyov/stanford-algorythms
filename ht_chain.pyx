"""
Hash Table in C using Chaining with Dynamic Arrays
"""

from libc.stdlib cimport malloc, free, calloc, rand
from array_c cimport array_c, create_arr, free_arr, push_back_arr, isin_arr, count_arr, py2arr, print_array

import cython
from cython.parallel import prange
from time import time

DEF ITEMS_PER_BUCKET = 2


""" Hash function h(x) = x mod n, where n is prime """


cdef hashtable* make_hashtable(size_t n):
    cdef:
        hashtable * h = <hashtable*>malloc(sizeof(hashtable))
    h.items = <array_c**>calloc(n, sizeof(array_c*))
    h.capacity = n
    h.size = 0
    return h


cdef void free_hashtable(hashtable* h):
    cdef:
        size_t i
        array_c* a
    for i in range(h.capacity):
        a = h.items[i]
        if a:
            free_arr(a)
    free(h.items)
    free(h)


cdef void insert(hashtable* h, size_t x):
    cdef:
        size_t i = 0
        size_t idx = hfunc(h, x)
        array_c* a = h.items[idx]

    if not a:
        a = create_arr(ITEMS_PER_BUCKET)
        h.items[idx] = a

    push_back_arr(a, x)
    h.size += 1

cdef size_t lookup_count(hashtable* h, size_t x):
    cdef:
        size_t i = 0
        size_t idx = hfunc(h, x) # hash(x)
        array_c* a = h.items[idx]

    if not a:
        return 0
    return count_arr(a, x)

cdef size_t twosum_ex(hashtable* h, array_c* a, long long int t) nogil:
    """
    Test for distinct pairs in hashtable, such that x + y = t
    :param h: hashtable
    :param a: array with all values in hashtable
    :param t: target sum
    :return: 1 if found, otherwise 0 
    """
    cdef:
        size_t i
        size_t cnt = 0
        long long int x
    for i in range(a.size):
        x = a.items[i]
        if x == t - x:
            continue
        if lookup(h, t - x):
            return 1
    return 0

cdef size_t twosum(hashtable* h, long long int t) nogil:
    """
    Test for distinct pairs in hashtable, such that x + y = t    
    Assuming that duplicates are filtered
    """
    cdef:
        size_t i, j
        size_t cnt = 0
        long long int x
        array_c* a
    for i in range(h.capacity):
        a = h.items[i]
        if a:
            for j in range(a.size):
                x = a.items[j]
                if lookup(h, t - x):
                    return 1
    return 0


cdef array_c* read_file():
    with open("2sum.txt", "r") as f:
        py_list = [int(x) for x in f.readlines()]
    cdef:
        size_t i
        array_c* a = create_arr(len(py_list))
    for i in range(len(py_list)):
        push_back_arr(a, <long long int>py_list[i])
    return a


def twosum_c(verbose=True):
    if verbose:
        print()
    start = time()
    cdef:
        int i
        size_t cnt = 0
        long long int x, t
        array_c* a = read_file()
        hashtable* h = make_hashtable(1313797)
        # hashtable* h = make_hashtable(2002823)
    if verbose:
        print(f"read_file: {time() - start:.2f}s")

    start = time()
    for i in range(a.size):
        x = a.items[i]
        if not lookup(h, x):
            insert(h, x)
    if verbose:
        print(f"make_hashtable: {time() - start:.2f}s")

    start = time()
    for i in prange(20001, nogil=True):
        t = -10000 + i
        cnt += twosum(h, t)

    # for i in range(64):
    #     t = -10000 + i
    #     # cnt += twosum_ex(h, a, t)
    #     cnt += twosum(h, t)
    if verbose:
        print(f"twosum: {time() - start:.2f}s")

    print(cnt)

    free_hashtable(h)
    free_arr(a)


def twosum_py():
    hash_dict = {}
    with open("2sum.txt", "r") as f:
        for line in f.readlines():
            hash_dict[int(line)] = 0

    cdef:
        long long int x
        long long int t = 0
        size_t cnt = 0

    for t in range(10):
        for x in hash_dict:
            if hash_dict.get(t - x):
                cnt += 1
    # print(len(hash_dict))
    # print(cnt)

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

import numpy as np


def test_read():
    cdef:
        array_c* a = read_file()
    assert a.items[0] == 68037543430
    assert <long long int>a.items[1] == -21123414637
    free_arr(a)


def test_make():
    cdef:
        size_t x = rand()
        hashtable* h = make_hashtable(7)
    insert(h, x)
    assert lookup(h, x)
    free_hashtable(h)

def test_lookup_cnt():
    cdef:
        size_t x = rand()
        hashtable* h = make_hashtable(7)

    assert lookup_count(h, x) == 0
    insert(h, x)
    assert lookup_count(h, x) == 1
    insert(h, x)
    assert lookup_count(h, x) == 2
    free_hashtable(h)

def test_lookup_rnd():
    cdef:
        size_t i
        size_t n = 100000
        hashtable* h = make_hashtable(102197)
        long long int [:] arr = np.random.randint(-100000, 100000, n, dtype=np.int64)
    for i in range(n):
        insert(h, arr[i])
    for i in range(n):
        assert lookup(h, arr[i])
    for i in range(n):
        assert lookup_count(h, arr[i]) > 0

    free_hashtable(h)

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
