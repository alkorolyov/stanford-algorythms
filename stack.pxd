# cython: language_level=3
ctypedef struct stack_c:
    size_t      capacity
    size_t      top
    size_t*     items

cdef:
    stack_c* create_stack(size_t n)
    bint is_empty_s(stack_c* s)
    bint is_full_s(stack_c* s)
    size_t size_s(stack_c* s)
    void push(stack_c* s, size_t x)
    size_t pop(stack_c* s)
    size_t peek(stack_c* s)
    void print_stack(stack_c* s)
    void free_stack(stack_c* s)