#cython: language_level=3

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from array_c cimport array_c, py2arr
cimport numpy as cnp
from numpy cimport PyArray_DIMS, PyArray_DATA, npy_intp, ndarray
cnp.import_array()

from utils import print_func_name
from time import time
import numpy as np
""" ################## Heap in C ######################### """

cdef heap_c* create_heap(size_t n):
    cdef heap_c* h = <heap_c*>PyMem_Malloc(sizeof(heap_c))
    if h == NULL: exit(1)

    h.items = <size_t*>PyMem_Malloc(n * sizeof(size_t))
    if h.items == NULL: exit(1)

    h.capacity = n
    h.size = 0
    return h


cdef inline void resize_heap(heap_c* h):
    h.capacity *= 2
    h.items = <size_t*>PyMem_Realloc(h.items, h.capacity * sizeof(size_t))
    if h.items == NULL: exit(1)

cdef void free_heap(heap_c* h):
    PyMem_Free(h.items)
    PyMem_Free(h)

cdef void print_heap(heap_c* h, size_t i=0, str indent="", bint last=False):
    cdef:
        size_t j, n, l, r

    label = f"{i}: [{h.items[i]}]"

    if i == 0:
        # print("heap size:", h.size)
        print(label)
    elif last:
        print(indent + "└╴" + label)
        indent += "  "
    else:
        print(indent + "├╴" + label)
        indent += "│ "

    n = get_child_cnt(h.size, i)

    for j in range(n):
        print_heap(h, get_children(h.size, i)[j], indent, j == n - 1)


cdef inline size_t _bubble_up(heap_c* h, size_t i):
    """
    Bubble i-th item up. Tries to swap with parent if 
    item's value is smaller. Otherwise returns -1.
    :param h: 
    :param i: index, zero terminated
    :return: parent index or -1
    """
    cdef:
        size_t p_idx

    # root reached
    if i == 0:
        return -1

    p_idx = get_parent_h(i)

    if h.items[i] < h.items[p_idx]:
        _swap(h.items, p_idx, i)
        return p_idx

    return -1

cdef inline size_t _min_child(heap_c* h, size_t l, size_t r):
    """
    Finds child with min value.
    :param h: heap
    :param l: left child idx
    :param r: right child idx
    :return: child idx with minimum value
    """
    if r == -1:
        return l
    if h.items[l] < h.items[r]:
        return l
    else:
        return r

cdef inline size_t _bubble_down(heap_c* h, size_t i):
    """
    Bubbles i-th item up, by swapping item with min
    out of two children, if it is smaller. Otherwise -1
    
    :param h: pointer to C heap
    :param i: index 
    :return: new index or -1
    """
    cdef:
        size_t l, r, min_idx

    l, r = get_children(h.size, i)

    if l == -1:
        return -1

    min_idx = _min_child(h, l, r)

    if h.items[i] > h.items[min_idx]:
        _swap(h.items, min_idx, i)
        return min_idx
    else:
        return -1


cdef void push_heap(heap_c* h, size_t x):
    if is_full_h(h):
        resize_heap(h)

    cdef size_t i = h.size

    h.items[i] = x
    h.size += 1

    while i != -1:
        i = _bubble_up(h, i)


cdef size_t pop_heap(heap_c* h):
    if is_empty_h(h):
        print("heap is empty")
        exit(1)

    cdef:
        size_t min_itm = h.items[0]
        size_t i = 0

    h.size -= 1
    if h.size == 0:
        return min_itm

    _swap(h.items, 0, h.size)
    while i != -1:
        i = _bubble_down(h, i)
    return min_itm

cdef void heapify(array_c* a):
    """
    Heapify array in place.
    :param a: input array pointer 
    :return: void
    """
    cdef:
        size_t i = a.size - 1
        size_t j
    while i < a.size:
        j = i
        while j != -1:
            j = _bubble_up(<heap_c*>a, j)
        i -= 1



""" ################################################################ """
""" ######################### TIMING ########################### """
""" ################################################################ """

cdef (size_t*, size_t) read_numpy(ndarray[unsigned long long, ndim=1] arr):
    cdef:
        npy_intp    *dims
        size_t      *data
    if arr.flags['C_CONTIGUOUS']:
        dims = PyArray_DIMS(arr)
        data = <size_t*>PyArray_DATA(arr)
        return data, <size_t>dims[0]
    else:
        print("numpy array is not C-contiguous")
        exit(1)

def time_log2():
    print_func_name()
    cdef:
        size_t*   data
        size_t    size
        size_t i
        size_t j = 0

    n = int(5e7)
    arr = np.random.randint(n // 2, size=n, dtype=np.uint64)
    data, size = read_numpy(arr)

    start_time = time()
    for i in range(size):
        j += data[i]
    loop_time = time() - start_time

    start_time = time()
    for i in range(size):
        j += log2(data[i])
    print(f"log2_lzcnt(): {(time() - start_time - loop_time):.3f}s")

    start_time = time()
    for i in range(size):
        j += log2_loop(data[i])
    print(f"log2_loop(): {(time() - start_time - loop_time):.3f}s")

    return j

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_log2():
    print_func_name()

    assert log2(0) == 0
    assert log2(1) == 0
    assert log2(2) == 1
    assert log2(3) == 1
    assert log2(4) == 2
    assert log2(5) == 2
    assert log2(7) == 2
    assert log2(8) == 3
    assert log2(9) == 3
    assert log2(0xFFFFFFFFFFFFFFFF) == 63


def test_get_parent():

    # assert get_parent_h(0) == -1
    assert get_parent_h(1) == 0
    assert get_parent_h(2) == 0
    assert get_parent_h(3) == 1
    assert get_parent_h(4) == 1
    assert get_parent_h(5) == 2

    # for i in range(15):
    #     print("i:", i, "parent:",get_parent_h(i))

def test_get_children():
    print_func_name()
    assert _get_l_child(0) == 1
    assert _get_l_child(1) == 3
    assert _get_l_child(2) == 5
    assert _get_l_child(3) == 7
    assert _get_l_child(4) == 9
    assert _get_l_child(5) == 11

    # for i in range(6):
    #     print("i:", i, "child:",_get_l_child(i))


def test_create():
    print_func_name()
    cdef heap_c* h = create_heap(5)
    push_heap(h, 3)
    push_heap(h, 4)
    push_heap(h, 2)
    push_heap(h, 1)
    push_heap(h, 0)
    assert h.items[0] == 0
    assert h.size == 5
    # print_heap(h)
    free_heap(h)


def test_heapify():
    print_func_name()
    # py_l = [4, 2, 3, 1, 0]
    py_l = [21, 32, 48, 14, 99, 4, 5, 7, 8, 9]
    cdef array_c* a = py2arr(py_l)
    # print_heap(<heap_c *> a)
    heapify(a)
    # print_heap(<heap_c *> a)
    assert a.items[0] == min(py_l)
    free_heap(<heap_c*>a)

def test_resize():
    print_func_name()
    cdef heap_c* h = create_heap(1)
    push_heap(h, 3)
    push_heap(h, 4)
    assert h.capacity == 2
    push_heap(h, 2)
    assert h.capacity == 4
    push_heap(h, 1)
    push_heap(h, 0)
    assert h.capacity == 8
    free_heap(h)


def test_pop_heap():
    print_func_name()
    cdef heap_c* h = create_heap(8)
    push_heap(h, 1)
    push_heap(h, 4)
    push_heap(h, 3)
    push_heap(h, 5)
    push_heap(h, 6)
    push_heap(h, 7)
    push_heap(h, 2)

    # print_heap(h)
    assert h.items[0] == 1
    assert h.size == 7
    assert pop_heap(h) == 1
    assert h.items[0] == 2
    assert h.size == 6

    # print_heap(h)
    free_heap(h)

def test_heap_rnd():
    DEF n = 100

    cdef:
        size_t [:] a
        long long [:] idx
        size_t i, j, k
        heap_c* h = create_heap(n // 4)

    np.random.seed(4)

    for j in range(100):
        arr = np.random.randint(0, n, n, dtype=np.uint64)
        a = arr

        for i in range(a.shape[0]):
            push_heap(h, a[i])

        idx = np.argsort(arr)
        for i in range(h.size):
            k = idx[i]
            assert a[k] == pop_heap(h)
        assert is_empty_h(h)


def test_heapify_rnd():
    print_func_name()
    DEF n = 10
    cdef:
        size_t [:] a_view
        size_t i, j, arr_min
        heap_c* h = create_heap(n)

    for j in range(10):
        arr = np.random.randint(0, 2 * n, n, dtype=np.uint64)
        a_view = arr

        for i in range(n):
            h.items[i] = a_view[i]
        h.size = n

        heapify(<array_c*>h)

        if h.items[0] != np.min(arr):
            print_heap(h)
            print(arr)
            print(np.min(arr))

        assert h.items[0] == np.min(arr)

        pop_heap(h)

        if h.items[0] != np.partition(arr, 1)[1]:
            print_heap(h)
            print(arr)
            print(np.partition(arr, 1)[1])

        assert h.items[0] == np.partition(arr, 1)[1]

        h.size = 0

def test_print_tree():
    print_func_name()
    cdef heap_c* h = create_heap(4)
    push_heap(h, 21)
    push_heap(h, 32)
    push_heap(h, 48)
    push_heap(h, 14)
    push_heap(h, 99)
    push_heap(h, 4)
    push_heap(h, 5)
    push_heap(h, 7)
    push_heap(h, 8)
    push_heap(h, 9)
    print_heap(h)
    free_heap(h)