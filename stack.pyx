# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

from libc.stdlib cimport malloc, free, EXIT_FAILURE
from utils import print_func_name
import numpy as np

""" ###################### Stack in C ########################## """

ctypedef struct stack_c:
    size_t      maxsize
    size_t      top
    size_t*     items

cdef stack_c* create_stack(size_t n):
    cdef stack_c* s = <stack_c*> malloc(sizeof(stack_c))
    s.maxsize = n
    s.top = -1
    s.items = <size_t*>malloc(sizeof(size_t) * n)
    return s

cdef bint is_empty_s(stack_c* s):
    return s.top == -1

cdef bint is_full_s(stack_c* s):
    return s.top == (s.maxsize - 1)

cdef size_t size_s(stack_c* s):
    return s.top + 1

cdef void push(stack_c* s, size_t x):
    if is_full_s(s):
        print("stack full")
        exit(EXIT_FAILURE)
    else:
        s.top += 1
        s.items[s.top] = x

cdef size_t pop(stack_c* s):
    cdef size_t temp
    if is_empty_s(s):
        print("stack empty")
        exit(EXIT_FAILURE)
    else:
        temp = s.items[s.top]
        s.top -= 1
        return temp

cdef size_t peek(stack_c* s):
    return s.items[s.top]

cdef void print_stack(stack_c* s):
    cdef size_t n = size_s(s)

    print("[", end="")

    if is_empty_s(s):
        print("]")
        return

    for i in range(n - 1):
        print(s.items[i], end=", ")
    print(s.items[n - 1], end="]\n")

cdef void free_stack(stack_c* s):
    free(s.items)
    free(s)


""" ############################# UNIT TESTS ######################### """

def test_stack_push():
    print_func_name()
    cdef stack_c* s = create_stack(5)
    push(s, 1)
    push(s, 2)
    push(s, 3)
    assert pop(s) == 3
    assert pop(s) == 2
    assert pop(s) == 1
    free_stack(s)

def test_stack_empty():
    print_func_name()
    cdef stack_c* s = create_stack(5)
    assert is_empty_s(s)
    push(s, 1)
    pop(s)
    assert is_empty_s(s)
    free_stack(s)

def test_stack_full():
    print_func_name()
    cdef stack_c* s = create_stack(1)
    assert is_empty_s(s)
    push(s, 0)
    assert is_full_s(s)
    free_stack(s)

def test_stack_size():
    print_func_name()
    cdef stack_c* s = create_stack(5)
    push(s, 1)
    assert size_s(s) == 1
    push(s, 1)
    assert size_s(s) == 2
    push(s, 1)
    assert size_s(s) == 3
    pop(s)
    assert size_s(s) == 2
    pop(s)
    assert size_s(s) == 1
    free_stack(s)


def test_stack_random():
    print_func_name()
    DEF size = 1000
    cdef size_t i, j
    cdef stack_c* s = <stack_c*>create_stack(size)

    for j in range(100):
        a = np.random.randint(0, 1000, size)
        for i in range(size):
            push(s, a[i])
        for i in range(size):
            assert pop(s) == a[size - 1 - i]

    free_stack(s)

