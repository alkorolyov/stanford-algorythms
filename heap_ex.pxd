# cython: language_level=3

ctypedef struct item:
    size_t  id
    size_t  val

ctypedef struct heap_ex:
    size_t  capacity
    size_t  size
    item*     items

cdef inline void _swap_el(item* a, size_t i, size_t j):
    cdef item tmp
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp

cdef:
    heap_ex * create_heap(size_t n)
    void free_heap(heap_ex * h)
    bint is_empty_h(heap_ex * h)
    bint isin_h(heap_ex * h, size_t id)
    size_t find_h(heap_ex * h, size_t id, size_t start=*)
    void push_heap(heap_ex* h, size_t id, size_t val)
    item pop_heap(heap_ex * h)
    void replace_h(heap_ex * h, size_t idx, size_t val)
    void print_heap(heap_ex * h, size_t i=*, str indent=*, bint last=*)



