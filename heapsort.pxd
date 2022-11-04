cdef inline size_t _get_child(size_t i):
    """ Return left child idx in heap. [right] = [left + 1] """
    # idx * 2
    return (i << 1) + 1


cdef inline size_t _get_parent(size_t i):
    """ Return parent idx in heap """
    # idx // 2
    return ((i + 1) >> 1) - 1


cdef inline size_t _max_idx(double* h, size_t l):
    """
    Finds child idx with max value.
    :param h: heap array pointer
    :param l: left child idx
    :return: child idx with minimum value
    """
    if h[l] > h[l + 1]:
        return l
    else:
        return l + 1


cdef inline size_t _min_idx(double* h, size_t l):
    """
    Finds child idx with min value.
    :param h: heap array pointer
    :param l: left child idx
    :return: child idx with minimum value
    """
    if h[l] < h[l + 1]:
        return l
    else:
        return l + 1

cdef:
    void hsort_c(double* a, size_t n)