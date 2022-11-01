

from c_utils cimport err_exit
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from heap_c cimport get_child_cnt, get_children, get_parent_h, _get_l_child
from utils import print_func_name, set_stdout, restore_stdout

import numpy as np

cdef heap_ex* create_heap(size_t n):
    cdef char* err_msg = "Heap create error"

    cdef heap_ex* h = <heap_ex*>PyMem_Malloc(sizeof(heap_ex))
    if h == NULL: err_exit(err_msg)

    h.items = <item*>PyMem_Malloc(n * sizeof(item))
    if h.items == NULL: err_exit(err_msg)

    h.idx = <size_t*>PyMem_Malloc(n * sizeof(size_t))
    if h.idx == NULL: err_exit(err_msg)

    cdef size_t i
    for i in range(n):
        h.idx[i] = -1

    h.capacity = n
    h.size = 0
    return h


cdef inline void resize_heap(heap_ex* h):
    cdef char* err_msg = "Heap resize error"

    h.capacity *= 2

    h.items = <item*>PyMem_Realloc(h.items, h.capacity * sizeof(item))
    if h.items == NULL: err_exit(err_msg)

    h.idx = <size_t *> PyMem_Realloc(h.idx, h.capacity * sizeof(size_t))
    if h.idx == NULL: err_exit(err_msg)


cdef void free_heap(heap_ex* h):
    PyMem_Free(h.idx)
    PyMem_Free(h.items)
    PyMem_Free(h)


# cdef inline bint is_full_h(heap_ex* h):
#     return h.size == h.capacity
#
#
# cdef inline bint is_empty_h(heap_ex* h):
#     return h.size == 0


cdef inline bint isin_h(heap_ex* h, size_t id):
    cdef  size_t i
    for i in range(h.size):
        if h.items[i].id == id:
            return True
    return False


cdef inline size_t _bubble_up(heap_ex* h, size_t i):
    """
    Bubble's i-th item up. Tries to swap with parent if 
    item's value is smaller. Returns new item index 
    (parent index) if goes up. Otherwise returns -1.
    :param h: C heap
    :param i: index (zero indexing)
    :return: new item index or -1
    """
    cdef:
        size_t p_idx

        item tmp
        item * a = h.items
        size_t id1
        size_t id2

    # root reached
    if i == 0:
        return -1

    # p_idx = get_parent_h(i)
    p_idx = ((i + 1) >> 1) - 1

    if h.items[i].val < h.items[p_idx].val:
        # _swap_el(h, p_idx, i)
        id1 = a[i].id
        id2 = a[p_idx].id

        h.idx[id1] = p_idx
        h.idx[id2] = i

        tmp = a[i]
        a[i] = a[p_idx]
        a[p_idx] = tmp

        return p_idx

    return -1


cdef inline size_t _bubble_down(heap_ex * h, size_t i):
    """
    Bubbles i-th item down, by swapping item with min
    out of two children, whichever smaller and returns new idx. 
    When no swap occurred returns -1.

    :param h: pointer to C heap
    :param i: index 
    :return: new index or -1
    """
    cdef:
        size_t l, r, min_idx
        size_t n = h.size - 1

        item tmp
        item * a = h.items
        size_t id1
        size_t id2


    # l = (i << 1) + 1 # left child of i
    # r = l + 1
    #
    # if l > n:
    #     # no children
    #     return -1
    # elif l == n:
    #     # only left child
    #     min_idx = l
    # else:
    #     # compare left and right
    #     if h.items[l].val < h.items[r].val:
    #         min_idx = l
    #     else:
    #         min_idx = r

    l, r = get_children(h.size, i)

    if l == -1:
        return -1

    min_idx = _min_child(h, l, r)

    if h.items[i].val > h.items[min_idx].val:
        _swap_el(h, min_idx, i)
        # id1 = a[i].id
        # id2 = a[min_idx].id
        #
        # h.idx[id1] = min_idx
        # h.idx[id2] = i
        #
        # tmp = a[i]
        # a[i] = a[min_idx]
        # a[min_idx] = tmp
        return min_idx

    return -1


cdef inline size_t _min_child(heap_ex* h, size_t l, size_t r):
    """
    Finds child with min value.
    :param h: heap
    :param l: left child idx
    :param r: right child idx
    :return: child idx with minimum value
    """
    if r == -1:
        return l
    if h.items[l].val < h.items[r].val:
        return l
    else:
        return r


cdef void replace_h(heap_ex* h, size_t idx, size_t val):
    cdef:
        size_t i = idx
    h.items[idx].val = val
    # try bubble up
    while i != -1:
        i = _bubble_up(h, i)
    # try bubble down
    i = idx
    while i != -1:
        i = _bubble_down(h, i)


cdef void push_heap(heap_ex* h, size_t id, size_t val):
    cdef size_t i = h.size
    if is_full_h(h):
        resize_heap(h)

    h.size += 1
    h.items[i].id = id
    h.items[i].val = val

    if id >= h.capacity:
        print("heap_ex id not mapped: out of bounds")
        exit(1)

    h.idx[id] = i

    while i != -1:
        i = _bubble_up(h, i)


cdef item pop_heap(heap_ex* h):
    if is_empty_h(h):
        print("Pop error: heap is empty")
        exit(1)

    cdef:
        item min = h.items[0]
        size_t i = 0

    h.idx[min.id] = -1

    h.size -= 1
    if h.size == 0:
        return min

    _swap_el(h, 0, h.size)
    while i != -1:
        i = _bubble_down(h, i)
    return min


cdef void print_heap(heap_ex* h, size_t i=0, str indent="", bint last=False):
    cdef:
        size_t j, n

    if is_empty_h(h):
        print("[]")
        return

    label = f"{h.items[i].id}: [{h.items[i].val}]"

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




""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_create_ex():
    print_func_name()
    cdef heap_ex* h = create_heap(3)
    h.items[0].id = 7
    h.items[0].val = 42
    free_heap(h)

def test_swap_ex():
    print_func_name()
    cdef heap_ex* h = create_heap(2)
    h.items[0].id = 1
    h.items[0].val = 42
    h.items[1].id = 0
    h.items[1].val = 13

    _swap_el(h, 0, 1)

    assert h.items[0].id == 0
    assert h.items[0].val == 13
    assert h.items[1].id == 1
    assert h.items[1].val == 42

    free_heap(h)

def test_print_ex():
    print_func_name()
    cdef heap_ex* h = create_heap(4)

    h.items[0].id = 7
    h.items[0].val = 42
    h.items[1].id = 0
    h.items[1].val = 13
    h.items[2].id = 3
    h.items[2].val = 9
    h.items[3].id = 1
    h.items[3].val = 37
    h.size = 4

    out = set_stdout()
    print_heap(h)
    free_heap(h)
    restore_stdout()
    assert out.getvalue() == "7: [42]\n├╴0: [13]\n│ └╴1: [37]\n└╴3: [9]\n"
    # print(out.getvalue(), end="")


def test_push_heap():
    print_func_name()
    cdef heap_ex * h = create_heap(5)
    push_heap(h, 3, 32)
    push_heap(h, 1, 57)
    push_heap(h, 0, 75)
    push_heap(h, 0, 12)
    push_heap(h, 2, 17)

    # print_heap(h)
    assert h.items[0].id == 0
    assert h.items[0].val == 12

    free_heap(h)

def test_isin():
    print_func_name()
    cdef heap_ex * h = create_heap(5)
    push_heap(h, 3, 32)
    push_heap(h, 1, 57)
    push_heap(h, 0, 75)
    push_heap(h, 0, 12)
    assert isin_h(h, 0)
    assert isin_h(h, 1)
    assert isin_h(h, 3)
    assert not isin_h(h, 4)
    free_heap(h)

def test_find():
    print_func_name()
    cdef heap_ex * h = create_heap(5)
    push_heap(h, 3, 32)
    push_heap(h, 1, 57)
    push_heap(h, 0, 75)
    push_heap(h, 0, 12)

    # print_heap(h)
    assert find_h(h, 0) == 0
    assert find_h(h, 0, 1) == 2
    assert find_h(h, 0, 3) == -1
    assert find_h(h, 1) == 3
    assert find_h(h, 3) == 1
    assert find_h(h, 2) == -1

    # while True:
    #     idx = find_h(h, 0, idx)
    #     if idx == -1:
    #         found = False
    #         break
    #     print(idx)
    #     idx += 1

    free_heap(h)


def test_resize():
    print_func_name()
    cdef heap_ex* h = create_heap(1)
    push_heap(h, 0, 3)
    push_heap(h, 1, 4)
    assert h.capacity == 2
    push_heap(h, 2, 2)
    assert h.capacity == 4
    push_heap(h, 3, 1)
    push_heap(h, 4, 0)
    assert h.capacity == 8
    free_heap(h)

def test_pop_heap():
    print_func_name()
    cdef heap_ex* h = create_heap(8)
    push_heap(h, 0, 1)
    push_heap(h, 1, 4)
    push_heap(h, 2, 3)
    push_heap(h, 3, 5)
    push_heap(h, 4, 6)
    push_heap(h, 5, 7)
    push_heap(h, 6, 2)

    # print_heap(h)
    assert h.items[0].val == 1
    assert h.size == 7
    assert pop_heap(h).val == 1
    assert h.items[0].val == 2
    assert h.size == 6

    # print_heap(h)
    free_heap(h)

def test_pop_heap_single():
    print_func_name()
    cdef heap_ex* h = create_heap(1)
    push_heap(h, 0, 98)
    push_heap(h, 1, 99)
    assert pop_heap(h).val == 98
    assert pop_heap(h).val == 99
    assert h.size == 0

    # print_heap(h)
    free_heap(h)



def test_replace():
    print_func_name()
    cdef heap_ex* h = create_heap(4)
    push_heap(h, 0, 1)
    push_heap(h, 1, 4)
    push_heap(h, 2, 3)
    push_heap(h, 3, 5)
    push_heap(h, 4, 6)
    push_heap(h, 5, 7)
    push_heap(h, 6, 2)

    replace_h(h, 3, 0)
    assert h.items[0].val == 0

    # print_heap(h)

    replace_h(h, 0, 10)
    assert h.items[3].val == 10

    # for i in range(h.size):
    #     print((i, h.items[i].val), end=", ")
    # print()

    # print_heap(h)
    free_heap(h)

def test_push_pop_rnd():
    print_func_name()
    DEF SIZE = 100

    cdef:
        size_t [:, :] a
        long long [:, :] idx
        size_t i, j, k
        item itm
        heap_ex* h = create_heap(SIZE)

    np.random.seed(4)

    for j in range(100):
        arr = np.random.randint(0, SIZE, (SIZE, 2), dtype=np.uint64)
        a = arr
        for i in range(arr.shape[0]):
            push_heap(h, arr[i, 0], arr[i, 1])

        idx = np.argsort(arr, axis=0)

        for i in range(h.size):
            itm = pop_heap(h)
            k = idx[i, 1] # idx in original array
            assert itm.val == a[k, 1]
        assert is_empty_h(h)

    free_heap(h)



