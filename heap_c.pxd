# cython: language_level=3

ctypedef struct heap_c:
    size_t  capacity
    size_t  size
    size_t* items

cdef:
    heap_c* create_heap(size_t n)
    void insert_h(heap_c * h, size_t x)
    size_t extract_min(heap_c * h)
    void free_heap(heap_c * h)
    void print_heap(heap_c * h, size_t i=*, str indent=*, bint last=*)