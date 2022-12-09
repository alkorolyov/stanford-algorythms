

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from utils import print_func_name, set_stdout, restore_stdout
import numpy as np

""" ###################### Stack in C ########################## """


cdef stack_c* create_stack(size_t capacity):
    cdef stack_c* s = <stack_c*> PyMem_Malloc(sizeof(stack_c))
    if s == NULL: exit(1)
    s.items = <size_t *> PyMem_Malloc(sizeof(size_t) * capacity)
    if s.items == NULL: exit(1)

    s.capacity = capacity
    s.top = -1
    return s

cdef void free_stack(stack_c* s):
    PyMem_Free(s.items)
    PyMem_Free(s)


# cdef void push(stack_c* s, size_t x):
#     if is_full_s(s):
#         print("stack push error: stack full")
#         exit(1)
#     else:
#         s.top += 1
#         s.items[s.top] = x
#
# cdef size_t pop(stack_c* s):
#     cdef size_t temp
#     if is_empty_s(s):
#         print("stack pop error: stack empty")
#         exit(1)
#     else:
#         temp = s.items[s.top]
#         s.top -= 1
#         return temp


cdef void print_stack(stack_c* s):
    cdef size_t n = size_s(s)

    if is_empty_s(s):
        print("[]")
        return

    print("[", end="")
    for i in range(n - 1):
        print(s.items[i], end=", ")
    print(s.items[n - 1], end="]\n")


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_push():
    
    cdef stack_c* s = create_stack(5)
    push(s, 1)
    push(s, 2)
    push(s, 3)
    assert pop(s) == 3
    assert pop(s) == 2
    assert pop(s) == 1
    free_stack(s)

def test_print():
    
    cdef stack_c* s = create_stack(5)
    push(s, 1)
    push(s, 2)
    push(s, 3)

    s_out = set_stdout()
    print_stack(s)
    out = s_out.getvalue()
    restore_stdout()

    assert out == '[1, 2, 3]\n'


    free_stack(s)


def test_empty():
    
    cdef stack_c* s = create_stack(5)
    assert is_empty_s(s)
    push(s, 1)
    pop(s)
    assert is_empty_s(s)
    free_stack(s)

def test_full():
    
    cdef stack_c* s = create_stack(1)
    assert is_empty_s(s)
    push(s, 0)
    assert is_full_s(s)
    free_stack(s)

def test_size():
    
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

def test_random():
    
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

