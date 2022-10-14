# cython: language_level=3
ctypedef struct array_c:
    size_t maxsize
    size_t size
    size_t* items

cdef:
    size_t max_arr(array_c * arr)
    array_c* list2arr(list py_list)
    array_c* create_arr(size_t maxsize)
    void resize_arr(array_c* arr)
    void free_arr(array_c* arr)
    void print_array(array_c* arr)