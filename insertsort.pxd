from sorting cimport _swap
from c_utils cimport swap

cdef inline void insertsort(double * a, size_t n):
    cdef:
        double x
        size_t i, j

    if a[0] > a[1]:
        _swap(a, 0, 1)

    i = 1
    while i < n:
        x = a[i]
        j = i - 1
        while j != -1 and a[j] > x:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = x
        i += 1

cdef inline void isort(double* lo, double* hi):
    cdef double x
    cdef double* pi
    cdef double* pj

    pi = &lo[0]
    while pi <= hi:
        x = pi[0]
        pj = pi - 1
        while pj >= lo and pj[0] > x:
            pj[1] = pj[0]
            pj -= 1
        pj[1] = x
        pi += 1


cdef inline void insert(double* a, size_t n):
    cdef size_t i, j
    x = a[n-1]

    for i in reversed(range(n - 1)):
        if x < a[i]:
            a[i+1] = a[i]
            if i == 0:
                a[i] = x
        else:
            a[i+1] = x
            break
