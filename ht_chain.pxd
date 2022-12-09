from array_c cimport array_c


ctypedef struct hashtable:
    size_t capacity
    size_t size
    array_c** items   # array of buckets


cdef inline size_t hfunc(hashtable* h, size_t x) nogil:
    return x % h.capacity


cdef inline bint isin_arr(array_c* arr, size_t val) nogil:
    cdef size_t i
    for i in range(arr.size):
        if arr.items[i] == val:
            return True
    return False


cdef inline bint lookup(hashtable* h, size_t x) nogil:
    cdef:
        # size_t idx = x % h.capacity
        size_t idx = hfunc(h, x)
        array_c* a = h.items[idx]

    if not a:
        return False
    return isin_arr(a, x)