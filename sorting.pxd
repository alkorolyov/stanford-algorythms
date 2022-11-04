cdef inline void _swap(double *a, size_t i, size_t j):
    cdef double t = a[i]
    a[i] = a[j]
    a[j] = t



cdef:
    size_t choose_p(double *arr, size_t n)
    size_t partition_c(double *arr, size_t n, size_t p_idx)
    size_t partition3_c(double *arr, size_t n, size_t p_idx)
    void qsort_c(double *arr, size_t n)
    void msort_c(double *arr, size_t n, double *buff)


