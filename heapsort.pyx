""" ================ Max heap in C ================== """

from c_utils cimport read_numpy
from sorting cimport _swap

import numpy as np
from utils import print_func_name



cdef void print_heap(double* h, size_t n, size_t i=0, str indent="", bint last=False):
    cdef:
        size_t l

    label = f"{i}: [{h[i]:.1f}]"

    if i == 0:
        # print("heap size:", h.size)
        print(label)
    elif last:
        print(indent + "└╴" + label)
        indent += "  "
    else:
        print(indent + "├╴" + label)
        indent += "│ "

    l = _get_child(i)

    if l > n - 1:
        return
    elif l == n - 1:
        print_heap(h, n, l, indent, last=True)
    else:
        print_heap(h, n, l + 1, indent, last=False)
        print_heap(h, n, l, indent, last=True)

cdef void assert_heap(double* h, size_t n):
    """ Test if array has heap property """
    cdef:
        size_t l
        size_t j = _get_parent(n - 1)

    while j != 0:
        l = _get_child(j)
        assert h[j] >= h[l]
        if l + 1 < n:
            assert h[j] >= h[l + 1]
        j -= 1


cdef inline void heapify(double* h, size_t n):
    """
    Heapify array in place. Get last non-leaf node (i),
    then heapify (bubble_down) each node up to root node.
    :param h: input array pointer
    :param n: size
    :return: void
    """
    cdef size_t i = _get_parent(n - 1)
    while i != -1:
        bubble_down(h, n, i)
        i -= 1


cdef inline void bubble_up(double* h, size_t i):
    cdef size_t j = i
    while j != -1:
        j = _bubble_up(h, j)


cdef inline size_t _bubble_up(double* h, size_t i):
    """
    Bubble i-th item up. Tries to swap with parent if 
    item's value is bigger and returns parent idx. 
    Otherwise returns -1.
    :param h: 
    :param i: index, zero terminated
    :return: parent index or -1
    """
    cdef size_t p_idx
    # root reached
    if i == 0:
        return -1
    p_idx = _get_parent(i)
    if h[i] > h[p_idx]:
        _swap(h, p_idx, i)
        return p_idx
    return -1


cdef inline void bubble_down(double* h, size_t n, size_t i):
    cdef size_t j = i
    while j != -1:
        j = _bubble_down(h, n, j)


cdef inline size_t _bubble_down(double * h, size_t n, size_t i):
    """
    Bubbles i-th item down, by swapping item with min
    out of two children, if it is bigger. Returns new
    swapped index if successful. Otherwise -1

    :param h: pointer to C heap
    :param i: index 
    :return: new index or -1
    """
    cdef:
        size_t l, max_idx

    l = _get_child(i)

    if l > n - 1:
        return -1
    elif l == n - 1:
        max_idx = l
    else:
        max_idx = _max_idx(h, l)

    if h[max_idx] > h[i]:
        _swap(h, max_idx, i)
        return max_idx
    else:
        return -1


cdef void hsort_c(double* a, size_t n):
    cdef size_t i, last = n - 1
    heapify(a, n)
    _swap(a, 0, n - 1)
    # print_heap(a, n)
    for i in range(1, n):
        # print(f"======== i {i} n {n - i} ========== ")
        bubble_down(a, n - i, 0)
        _swap(a, 0, n - 1 - i)
        # print_heap(a, n - i)

        # bubble_down(a + i, n - i, 0)


def hsort_py(arr):
    cdef:
        double* a
        size_t  n
    a, n = read_numpy(arr)
    hsort_c(a, n)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_heapify():
    print_func_name()
    cdef:
        size_t n = 100
        double* a
        size_t i
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        heapify(a, n)
        # print_heap(a, n)
        assert a[0] == np.max(arr)

def test_heap():
    print_func_name()
    cdef:
        size_t n = 100
        double* a
        size_t i
    # np.random.seed(2)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        # print_heap(a, n)
        heapify(a, n)
        # print_heap(a, n)
        assert_heap(a, n)

def test_hsort():
    print_func_name()
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
        # print(arr)
        hsort_c(a, n)
        for j in range(n):
            # if a[j] != a_mv[j]:
            #     print_heap(a, n)
            #     print(arr)
            #     print(np.sort(arr))

            assert a[j] == a_mv[j]
        # print(arr)



