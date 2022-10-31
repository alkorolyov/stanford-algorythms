# cython: language_level=3

ctypedef struct queue:
    size_t capacity
    size_t front
    size_t rear
    size_t* items

cdef:
    queue * create_queue(size_t capacity)
    void free_queue(queue * q)
    bint is_full_q(queue * q)
    bint is_empty_q(queue * q)
    void enqueue(queue * q, size_t x)
    size_t dequeue(queue * q)