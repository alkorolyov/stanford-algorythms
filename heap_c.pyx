#cython: language_level=3

from libc.stdlib cimport malloc, realloc, free, EXIT_FAILURE
from array_c cimport array_c, list2arr
cimport numpy as cnp
cnp.import_array()

from utils import print_func_name
from time import time
import numpy as np


cdef inline size_t log2_loop(size_t x) nogil:
    cdef:
        size_t i = 0
        size_t c = 1
    if x == 0:
        return 0
    while c <= x:
        c *= 2
        i += 1
    return i - 1

""" ################## Heap in C ######################### """

cdef inline (size_t, size_t) _get_level_idx(size_t idx):
    """
    Calculates level and index relative to this level.
    :param idx: zero indexed
    :return: level, relative index
    """
    cdef size_t n = _get_level(idx)
    return n, idx + 1 - (1ULL << n),

cdef heap_c* create_heap(size_t n):
    cdef:
        heap_c* h = <heap_c*>malloc(sizeof(heap_c))
    h.capacity = n
    h.size = 0
    h.items = <size_t*>malloc(n * sizeof(size_t))
    return h


cdef inline void resize_heap(heap_c* h):
    h.capacity *= 2
    h.items = <size_t*>realloc(h.items, h.capacity * sizeof(size_t))

cdef void free_heap(heap_c* h):
    free(h.items)
    free(h)

cdef void print_heap(heap_c* h, size_t i=0, str indent="", bint last=False):
    cdef:
        size_t j, n

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


cdef bint is_full_h(heap_c* h):
    return h.size == h.capacity

cdef bint is_empty_h(heap_c* h):
    return h.size == 0


cdef inline size_t _bubble_up(heap_c* h, size_t i):
    """
    Bubble i-th item up. Tries to swap with parent if 
    it is bigger. Otherwise returns 0.
    :param h: 
    :param i: index, zero terminated
    :return: parent index or 0
    """
    cdef:
        size_t p_idx = get_parent_h(i)
    if h.items[i] < h.items[p_idx]:
        _swap(h.items, p_idx, i)
        return p_idx
    else:
        return 0

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
        size_t l, r, min_c

    l, r = get_children(h.size, i)

    if l == -1:
        return -1

    min_c = _min_child(h, l, r)
    # print("min_c:", min_c)

    if h.items[i] > h.items[min_c]:
        _swap(h.items, min_c, i)
        return min_c
    else:
        return -1


cdef void insert_h(heap_c* h, size_t x):
    cdef size_t i = h.size
    if is_full_h(h):
        resize_heap(h)
    h.size += 1
    h.items[i] = x

    while i != 0:
        i = _bubble_up(h, i)


cdef size_t extract_min(heap_c* h):
    cdef:
        size_t min_val = h.items[0]
        size_t i = 0

    if is_empty_h(h):
        print("heap is empty")
        exit(EXIT_FAILURE)

    h.size -= 1
    _swap(h.items, 0, h.size)
    while i != -1:
        i = _bubble_down(h, i)
    return min_val

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
        while j != 0:
            j = _bubble_up(<heap_c*>a, j)
        i -= 1



""" ################################################################ """
""" ######################### TIMING ########################### """
""" ################################################################ """

cdef (size_t*, cnp.npy_intp*) read_numpy(cnp.ndarray[unsigned long long, ndim=1] arr):
    cdef:
        cnp.npy_intp    *dims
        size_t          *data
    if cnp.PyArray_Check(arr):
        dims = cnp.PyArray_DIMS(arr)
        data = <size_t*>cnp.PyArray_DATA(arr)
    return data, dims

def time_log2():
    print_func_name()
    cdef:
        size_t* data
        cnp.npy_intp* dims
        size_t i
        size_t j = 0

    n = int(5e7)
    arr = np.random.randint(n // 2, size=n, dtype=np.uint64)
    data, dims = read_numpy(arr)

    start_time = time()
    for i in range(dims[0]):
        j += data[i]
    loop_time = time() - start_time

    start_time = time()
    for i in range(dims[0]):
        j += log2(data[i])
    print(f"log2_lzcnt(): {(time() - start_time - loop_time):.3f}s")

    start_time = time()
    for i in range(dims[0]):
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


def test_get_level_idx():
    print_func_name()
    assert _get_level_idx(0) == (0, 0)
    assert _get_level_idx(1) == (1, 0)
    assert _get_level_idx(2) == (1, 1)
    assert _get_level_idx(3) == (2, 0)

def test_get_parent():

    assert get_parent_h(0) == -1
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
    insert_h(h, 3)
    insert_h(h, 4)
    insert_h(h, 2)
    insert_h(h, 1)
    insert_h(h, 0)
    assert h.items[0] == 0
    assert h.size == 5
    # print_heap(h)
    free_heap(h)


def test_heapify():
    print_func_name()
    py_l = [4, 2, 3, 1, 0]
    # py_l = [21, 32, 48, 14, 99, 4, 5, 7, 8, 9]
    cdef array_c* a = list2arr(py_l)
    # print_heap(<heap_c *> a)
    heapify(a)
    # print_heap(<heap_c *> a)
    assert a.items[0] == min(py_l)
    free_heap(<heap_c*>a)

def test_resize():
    print_func_name()
    cdef heap_c* h = create_heap(1)
    insert_h(h, 3)
    insert_h(h, 4)
    assert h.capacity == 2
    insert_h(h, 2)
    assert h.capacity == 4
    insert_h(h, 1)
    insert_h(h, 0)
    assert h.capacity == 8
    free_heap(h)


def test_extract_min():
    print_func_name()
    cdef heap_c* h = create_heap(8)
    insert_h(h, 1)
    insert_h(h, 4)
    insert_h(h, 3)
    insert_h(h, 5)
    insert_h(h, 6)
    insert_h(h, 7)
    insert_h(h, 2)

    # print_heap(h)
    assert h.items[0] == 1
    assert h.size == 7
    assert extract_min(h) == 1
    assert h.items[0] == 2
    assert h.size == 6

    # print_heap(h)
    free_heap(h)

def test_heap_rnd():
    print_func_name()
    DEF n = 100

    cdef:
        size_t [:] a
        size_t i, j, arr_min
        heap_c* h = create_heap(n // 4)

    for j in range(100):
        arr = np.random.randint(0, 2 * n, n, dtype=np.uint64)
        a = arr
        for i in range(n):
            insert_h(h, a[i])
        assert h.items[0] == np.min(arr)
        extract_min(h)
        assert h.items[0] == np.partition(arr, 1)[1]
        h.size = 0

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

        extract_min(h)

        if h.items[0] != np.partition(arr, 1)[1]:
            print_heap(h)
            print(arr)
            print(np.partition(arr, 1)[1])

        assert h.items[0] == np.partition(arr, 1)[1]

        h.size = 0

def test_print_tree():
    print_func_name()
    cdef heap_c* h = create_heap(4)
    insert_h(h, 21)
    insert_h(h, 32)
    insert_h(h, 48)
    insert_h(h, 14)
    insert_h(h, 99)
    insert_h(h, 4)
    insert_h(h, 5)
    insert_h(h, 7)
    insert_h(h, 8)
    insert_h(h, 9)
    print_heap(h)
    free_heap(h)