from libc.stdlib cimport exit
from libc.stdio cimport printf

ctypedef struct hashtable:
    size_t capacity
    size_t size
    size_t* items   # array of buckets


cdef inline void swap(size_t* a, size_t* b) nogil:
    cdef size_t tmp = a[0]
    a[0] = b[0]
    b[0] = tmp


cdef inline size_t hfunc(hashtable* h, size_t x) nogil:
    return x % h.capacity


cdef inline void insert(hashtable* h, size_t x) nogil:
    cdef:
        size_t i = hfunc(h, x)
        size_t* pi = &h.items[i]
        size_t* hi = h.items + h.capacity - 1
    while pi[0]:
        pi += 1
    pi[0] = x
    h.size += 1


cdef inline size_t* search(hashtable* h, size_t x) nogil:
    cdef:
        size_t i = hfunc(h, x)
        size_t *pi = &h.items[i]
        size_t *hi = h.items + h.capacity - 1
    while pi[0]:
        if pi[0] == x:
            return pi
        pi += 1
    return NULL


cdef inline void delete(hashtable* h, size_t x) nogil:
    """
    Delete element from hash table, by detecting and fixing
    collisions in cluster of element. 
    i-th position of empty cell
    j-th current position to check
    k-th natural hash position of a[j] element (always k <= j)

    Cyclically move empty cell (i-th) upwards by swapping
    with current cell (j-th) until the end of cluster. If swapping
    will corrupt hash property, we skip to the next (j + 1) cell.    
    """

    cdef:
        size_t k
        size_t* pi = search(h, x)
        size_t* hi = h.items + h.capacity - 1
        size_t* pj

    if not pi:
        return

    pi[0] = 0
    h.size -= 1

    pj = pi + 1
    while True:
        if pj[0] == 0:
            return
        # skip cell if switching j-th position would corrupt cluster
        k = hfunc(h, pj[0])
        if pi < k + h.items:
            pj += 1
            continue
        # swap empty element
        swap(pi, pj)
        pi = pj # new empty element
        pj += 1