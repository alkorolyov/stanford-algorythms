# cython: language_level=3


ctypedef struct heap_c:
    size_t  capacity
    size_t  size
    size_t* items

cdef extern size_t _lzcnt_u64 (size_t x)
cdef extern size_t _lzcnt_u32 (size_t x)

cdef inline void _swap(size_t* a, size_t i, size_t j):
    cdef size_t tmp
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp

cdef inline size_t log2(size_t x):
    if x == 0:
        return 0
    return 63 - _lzcnt_u64(x)

cdef inline size_t _get_level(size_t idx):
    return log2(idx + 1)

cdef inline size_t get_parent_h(size_t i):
    if i == 0:
        return -1
    # idx // 2
    return ((i + 1) >> 1) - 1

cdef inline size_t _get_l_child(size_t i):
    # idx * 2
    return (i << 1) + 1

cdef inline (size_t, size_t) get_children(size_t h_size, size_t i):
    cdef size_t l_idx = _get_l_child(i)
    if l_idx > h_size - 1:
        return -1, -1
    elif l_idx == h_size - 1:
        return l_idx, -1
    else:
        return l_idx, l_idx + 1

cdef inline size_t get_child_cnt(size_t h_size, size_t i):
    cdef size_t l, r
    l, r = get_children(h_size, i)
    if l == -1:
        return 0
    if r == -1:
        return 1
    else:
        return 2

cdef:
    heap_c* create_heap(size_t n)
    void insert_h(heap_c * h, size_t x)
    size_t extract_min(heap_c * h)
    void free_heap(heap_c * h)
    void print_heap(heap_c * h, size_t i=*, str indent=*, bint last=*)