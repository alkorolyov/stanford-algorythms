# cython: language_level=3
ctypedef struct stack_c:
    size_t      maxsize
    size_t      top
    size_t*     items

cdef stack_c* create_stack(size_t n)
cdef bint is_empty_s(stack_c* s)
cdef bint is_full_s(stack_c* s)
cdef size_t size_s(stack_c* s)
cdef void push(stack_c* s, size_t x)
cdef size_t pop(stack_c* s)
cdef size_t peek(stack_c* s)
cdef void print_stack(stack_c* s)
cdef void free_stack(stack_c* s)