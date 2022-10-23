# cython: language_level=3
ctypedef struct array_c:
    size_t capacity
    size_t size
    size_t* items

cdef:
    array_c* list2arr(object py_obj)
    object arr2numpy(array_c * arr)
    array_c* create_arr(size_t n)
    array_c* create_arr_val(size_t n, size_t val)
    void free_arr(array_c * arr)
    void push_back_arr(array_c * arr, size_t val)
    void resize_arr(array_c* arr)
    size_t max_arr(array_c * arr)
    bint isin_arr(array_c * arr, size_t val)
    void reverse_arr(array_c * arr)
    void print_array(array_c* arr)
