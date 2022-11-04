from sorting cimport _swap

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

    # for size in range(2, n + 1):
    #     insert(a, size)


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
