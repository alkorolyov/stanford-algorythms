# cython: language_level=3
ctypedef struct array_c:
    size_t maxsize
    size_t* items
cdef array_c* create_arr(size_t maxsize)
cdef void resize_arr(array_c* arr)
cdef void free_arr(array_c* arr)