# cython: language_level=3

from libc.stdlib cimport malloc, free, EXIT_FAILURE
from utils import print_func_name, set_stdout, restore_stdout
import numpy as np

""" ################## Queue in C ######################### """

ctypedef struct queue:
    size_t maxsize
    size_t front
    size_t rear
    size_t* items

cdef queue* create_queue(size_t n):
    cdef queue* q = <queue*> malloc(sizeof(queue))
    q.front = 0
    q.rear = -1
    q.maxsize = n
    q.items = <size_t*> malloc(n * sizeof(size_t))
    return q

cdef void free_queue(queue* q):
    free(q.items)
    free(q)

cdef bint is_full_q(queue* q):
    return q.rear == q.maxsize - 1

cdef bint is_empty_q(queue* q):
    return q.front == q.rear + 1

cdef void enqueue(queue* q, size_t x):
    if not is_full_q(q):
        q.rear += 1
        q.items[q.rear] = x
    else:
        print("Enqueue error: queue is full")
        exit(EXIT_FAILURE)

cdef size_t dequeue(queue* q):
    cdef size_t x
    if not is_empty_q(q):
        x = q.items[q.front]
        q.front += 1
        return x
    else:
        print("Dequeue error: queue is empty")
        exit(EXIT_FAILURE)

cdef size_t size_q(queue* q):
    return q.rear + 1 - q.front

cdef void print_queue(queue* q):
    cdef:
        size_t i, idx
        size_t n = size_q(q)

    if is_empty_q(q):
        print("[]")
        return

    print("[", end="")
    # idx = q.front
    for i in range(q.front, q.front + n - 1):
        print(q.items[i], end=", ")
    i += 1
    print(q.items[i], end="]\n")

""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_enqueue():
    print_func_name()
    cdef queue* q = create_queue(3)
    enqueue(q, 1)
    enqueue(q, 2)
    enqueue(q, 3)
    assert dequeue(q) == 1
    assert dequeue(q) == 2
    assert dequeue(q) == 3
    free_queue(q)

def test_empty():
    print_func_name()
    cdef queue* q = create_queue(3)
    assert is_empty_q(q)
    enqueue(q, 1)
    dequeue(q)
    assert is_empty_q(q)

def test_full():
    print_func_name()
    cdef queue* q = create_queue(2)
    enqueue(q, 1)
    enqueue(q, 2)
    assert is_full_q(q)

def test_print():
    print_func_name()

    cdef queue* q = create_queue(3)
    enqueue(q, 1)
    enqueue(q, 2)
    enqueue(q, 3)

    s = set_stdout()
    print_queue(q)
    out = s.getvalue()
    restore_stdout()

    assert out == '[1, 2, 3]\n'

