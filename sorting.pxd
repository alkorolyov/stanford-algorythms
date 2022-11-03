
cdef:
    size_t choose_p(double *arr, size_t n)
    size_t partition_c(double *arr, size_t n, size_t p_idx)
    size_t partition3_c(double *arr, size_t n, size_t p_idx)
    void qsort_c(double *arr, size_t n)
    void msort_c(double *arr, size_t n, double *buff)


