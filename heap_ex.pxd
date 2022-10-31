# cython: language_level=3

ctypedef struct item:
    size_t  id
    size_t  val

ctypedef struct heap_ex:
    size_t      capacity
    size_t      size
    item*       items
    size_t*     idx     # crossreference for id, idx[id] - position of id in array
                        # id's should be mapped to the heap capacity - [0 .. capacity - 1]


cdef inline size_t find_h(heap_ex* h, size_t id, size_t start=0):
    cdef size_t i
    for i in range(start, h.size):
        if h.items[i].id == id:
            return i
    return -1


cdef inline void _swap_el(heap_ex* h, size_t i, size_t j):
    cdef:
        item tmp
        item* a = h.items
        size_t id1 = a[i].id
        size_t id2 = a[j].id

    if id1 < h.capacity and id2 < h.capacity:
        h.idx[id1] = j
        h.idx[id2] = i

    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp


cdef inline item peek_heap(heap_ex* h):
    return h.items[0]

cdef:
    heap_ex * create_heap(size_t n)
    void free_heap(heap_ex * h)
    bint is_empty_h(heap_ex * h)
    bint isin_h(heap_ex * h, size_t id)
    void push_heap(heap_ex* h, size_t id, size_t val)
    item pop_heap(heap_ex * h)
    void replace_h(heap_ex * h, size_t idx, size_t val)
    void print_heap(heap_ex * h, size_t i=*, str indent=*, bint last=*)



