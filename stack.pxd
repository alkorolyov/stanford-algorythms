ctypedef struct stack_c:
    size_t      capacity
    size_t      top
    size_t*     items

cdef inline bint is_empty_s(stack_c* s):
    return s.top == -1

cdef inline bint is_full_s(stack_c* s):
    return s.top == (s.capacity - 1)

cdef inline size_t size_s(stack_c* s):
    return s.top + 1

cdef inline void push(stack_c* s, size_t x):
    if is_full_s(s):
        print("stack overflow")
        exit(1)
    s.top += 1
    s.items[s.top] = x

cdef inline size_t pop(stack_c* s):
    cdef size_t tmp
    tmp = s.items[s.top]
    s.top -= 1
    return tmp

cdef inline size_t peek(stack_c* s):
    return s.items[s.top]


cdef:
    stack_c* create_stack(size_t n)
    void print_stack(stack_c* s)
    void free_stack(stack_c* s)
    # bint is_empty_s(stack_c* s)
    # bint is_full_s(stack_c* s)
    # size_t size_s(stack_c* s)
    # void push(stack_c* s, size_t x)
    # size_t pop(stack_c* s)
    # size_t peek(stack_c* s)
