
cdef inline void _swap(double *a, size_t i, size_t j):
    cdef double t = a[i]
    a[i] = a[j]
    a[j] = t

cdef inline size_t median3(double *a, size_t n):
    """ 
    Choose pivot by median of 3 approach:
    median of first, middle and last elements
    :param arr: input array 
    :param n: array size
    :return: pivot index

    """
    cdef:
        size_t mid = n >> 1 # n // 2
        size_t hi = n - 1

    if a[0] < a[mid]:
        # a[0] < a[mid]
        if a[hi] > a[mid]:
            # median is a[mid]
            return mid
        elif a[hi] < a[0]:
            # median is a[0]
            return 0
        else:
            # median is a[hi]
            return hi
    else:
        # a[mid] <= a[0]
        if a[hi] > a[0]:
            # median is a[0]
            return 0
        elif a[hi] < a[mid]:
            # median is a[mid]
            return mid
        else:
            # median is a[hi]
            return hi


cdef:
    size_t partition_c(double *arr, size_t n, size_t p_idx)
    size_t partition3_c(double *arr, size_t n, size_t p_idx)
    void qsort_c(double *arr, size_t n)
    void msort_c(double *arr, size_t n, double *buff)


