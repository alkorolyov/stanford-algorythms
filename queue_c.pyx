

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from utils import print_func_name, set_stdout, restore_stdout


""" ################## Queue in C ######################### """

cdef queue* create_queue(size_t n):
    cdef queue* q = <queue*> PyMem_Malloc(sizeof(queue))
    if q == NULL: exit(1)
    q.items = <size_t *> PyMem_Malloc(n * sizeof(size_t))
    if q.items == NULL: exit(1)

    q.front = 0
    q.rear = -1
    q.capacity = n

    return q

cdef void free_queue(queue* q):
    PyMem_Free(q.items)
    PyMem_Free(q)

cdef inline bint is_full_q(queue* q):
    return q.rear == q.capacity - 1

cdef inline bint is_empty_q(queue* q):
    return q.front == q.rear + 1

cdef void enqueue(queue* q, size_t x):
    if not is_full_q(q):
        q.rear += 1
        q.items[q.rear] = x
    else:
        print("Enqueue error: queue is full")
        exit(1)

cdef size_t dequeue(queue* q):
    cdef size_t x
    if not is_empty_q(q):
        x = q.items[q.front]
        q.front += 1
        return x
    else:
        print("Dequeue error: queue is empty")
        exit(1)

cdef inline size_t size_q(queue* q):
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
    
    cdef queue* q = create_queue(3)
    enqueue(q, 1)
    enqueue(q, 2)
    enqueue(q, 3)
    assert dequeue(q) == 1
    assert dequeue(q) == 2
    assert dequeue(q) == 3
    free_queue(q)

def test_empty():
    
    cdef queue* q = create_queue(3)
    assert is_empty_q(q)
    enqueue(q, 1)
    dequeue(q)
    assert is_empty_q(q)

def test_full():
    
    cdef queue* q = create_queue(2)
    enqueue(q, 1)
    enqueue(q, 2)
    assert is_full_q(q)

def test_print():
    

    cdef queue* q = create_queue(3)
    enqueue(q, 1)
    enqueue(q, 2)
    enqueue(q, 3)

    s = set_stdout()
    print_queue(q)
    out = s.getvalue()
    restore_stdout()

    assert out == '[1, 2, 3]\n'

