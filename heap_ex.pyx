# cython: language_level=3

from libc.stdlib cimport malloc, realloc, free, EXIT_FAILURE
from utils import print_func_name
from heap_c cimport get_child_cnt, get_children, get_parent_h


cdef heap_ex* create_heap(size_t n):
    cdef:
        heap_ex* h = <heap_ex*>malloc(sizeof(heap_ex))
    h.capacity = n
    h.size = 0
    h.items = <item*>malloc(n * sizeof(item))
    return h

cdef inline void resize_heap(heap_ex* h):
    h.capacity *= 2
    h.items = <item*>realloc(h.items, h.capacity * sizeof(item))

cdef void free_heap(heap_ex* h):
    free(h.items)
    free(h)

cdef bint is_full_h(heap_ex* h):
    return h.size == h.capacity


cdef inline bint is_empty_h(heap_ex* h):
    return h.size == 0

cdef inline bint isin_h(heap_ex* h, size_t id):
    cdef  size_t i
    for i in range(h.size):
        if h.items[i].id == id:
            return True
    return False

cdef size_t find_h(heap_ex* h, size_t id, size_t start=0):
    cdef  size_t i
    for i in range(start, h.size):
        if h.items[i].id == id:
            return i
    return -1


cdef inline size_t _bubble_up(heap_ex* h, size_t i):
    """
    Bubble i-th item up. Tries to swap with parent if 
    it's value is bigger. Otherwise returns 0.
    :param h: 
    :param i: index, zero terminated
    :return: parent index or 0
    """
    cdef:
        size_t p_idx = get_parent_h(i)
    if h.items[i].val < h.items[p_idx].val:
        _swap_el(h.items, p_idx, i)
        return p_idx
    else:
        return 0

cdef inline size_t _bubble_down(heap_ex * h, size_t i):
    """
    Bubbles i-th item down, by swapping item with min
    out of two children, whichever smaller. Otherwise -1

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
    # print("min_idx:", min_idx)

    if h.items[i].val > h.items[min_idx].val:
        _swap_el(h.items, min_idx, i)
        return min_idx
    else:
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
    while i != 0:
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

    while i != 0:
        i = _bubble_up(h, i)


cdef item pop_heap(heap_ex* h):
    if is_empty_h(h):
        print("heap is empty")
        exit(EXIT_FAILURE)

    cdef:
        item min_el = h.items[0]
        size_t i = 0

    h.size -= 1
    _swap_el(h.items, 0, h.size)
    while i != -1:
        i = _bubble_down(h, i)
    return min_el

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
    h.items[0].id = 7
    h.items[0].val = 42
    h.items[1].id = 0
    h.items[1].val = 13

    _swap_el(h.items, 0, 1)

    assert h.items[0].id == 0
    assert h.items[0].val == 13
    assert h.items[1].id == 7
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
    print_heap(h)
    free_heap(h)

def test_push_heap():
    print_func_name()
    cdef heap_ex * h = create_heap(5)
    push_heap(h, 3, 32)
    push_heap(h, 1, 57)
    push_heap(h, 0, 75)
    push_heap(h, 0, 12)
    push_heap(h, 2, 17)
    assert h.items[0].id == 0
    assert h.items[0].val == 12
    # print_heap(h)
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
    cdef size_t idx
    assert find_h(h, 0) == 0
    assert find_h(h, 0, 1) == 2
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
    push_heap(h, 1, 3)
    push_heap(h, 2, 4)
    assert h.capacity == 2
    push_heap(h, 3, 2)
    assert h.capacity == 4
    push_heap(h, 4, 1)
    push_heap(h, 5, 0)
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



