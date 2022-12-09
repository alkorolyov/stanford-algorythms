from libc.stdlib cimport RAND_MAX
from c_utils cimport frand, frand32, read_numpy, log2, print_arr, swap, \
    med3, _med3, imed3, med3_sse
from insertsort cimport insertsort, isort
from stack cimport stack_c, create_stack, push, pop, size_s, is_empty_s, free_stack

from utils import print_func_name
import numpy as np

cdef max_depth = 0
""" ================= QuickSort in C ================== """

""" QuickSort stack version """
cdef void qsort_s(stack_c* stack):
    cdef:
        size_t n
        double* lo
        double* hi
        double* mid
        double p
        double* pp

    while not is_empty_s(stack):
        hi = <double*>pop(stack)
        lo = <double*>pop(stack)
        n = hi - lo

        # base cases
        if n == 0:
            continue
        if n == 1:
            if lo[0] > hi[0]:
                swap(lo, hi)
            continue

        # insertsort on small casses
        if n < 16:
            isort(lo, hi)
            continue

        # choose pivot
        p = med3(lo, hi)
        pp = part_h(lo, hi, p)

        # qsort(lo, pp)
        push(stack, <size_t>lo)
        push(stack, <size_t>pp)

        # qsort(pp + 1, hi)
        push(stack, <size_t>(pp + 1))
        push(stack, <size_t>hi)

cdef void qsort(double* lo, double* hi, size_t depth=0):
    cdef:
        size_t n = hi - lo
        double p
        double* pp
        double* mid
        # size_t i, j, p_idx, idx, delta

    # base cases
    if n == 0:
        return
    if n == 1:
        if lo[0] > hi[0]:
            swap(lo, hi)
        return

    # insertsort on small casses
    if n < 16:
        isort(lo, hi)
        return

    # global max_depth
    # if depth == max_depth:
    #     hsort_c(lo, n + 1)
    #     return

    # choose pivots
    # if n > RAND_MAX:
    #     p_idx = frand32() % n
    # else:
    #     p_idx = frand() % n
    # p_idx = frand32() % n
    # p_idx = frand() % n
    # p_idx = median3(lo, n)


    # # Lomuto's partition
    # pp = part_l(lo, hi, &lo[p_idx])
    # qsort(lo, pp - 1)
    # qsort(pp + 1, hi)

    # Hoare's partition
    # p = lo[p_idx]
    p = med3(lo, hi)
    # p = med3(lo, hi)
    # p = imed3(lo, hi)
    # p = _med3(lo, hi)[0]

    # p = max(min(lo[0], mid[0]), min(max(lo[0], mid[0]), hi[0]))

    # median of 3: 3 cmps max, 0 swaps
    # mid = lo + (n >> 1)  # (hi + lo) // 2
    # if lo[0] < mid[0]:
    #     # a[0] < a[mid]
    #     if hi[0] > mid[0]:
    #         # median is a[mid]
    #         p = mid[0]
    #     elif hi[0] < lo[0]:
    #         # median is a[0]
    #         p = lo[0]
    #     else:
    #         # median is a[hi]
    #         p = hi[0]
    # else:
    #     # a[mid] <= a[0]
    #     if hi[0] > lo[0]:
    #         # median is a[0]
    #         p = lo[0]
    #     elif hi[0] < mid[0]:
    #         # median is a[mid]
    #         p = mid[0]
    #     else:
    #         # median is a[hi]
    #         p = hi[0]

    # # median of 3: 3 cmps, max 3 swaps
    # mid = lo + (n >> 1) # (hi + lo) // 2
    # if mid[0] < lo[0]:
    #     swap_c(lo, mid)
    # if hi[0] < lo[0]:
    #     swap_c(lo, hi)
    # if mid[0] < hi[0]:
    #     swap_c(mid, hi)
    # p = hi[0]

    # pp = part_h(lo, hi, p)
    pp = part_h(lo, hi, p)

    qsort(lo, pp)
    qsort(pp + 1, hi)

    # if pp < mid:
    #     qsort(lo, pp)
    #     qsort(pp + 1, hi, depth + 1)
    # else:
    #     qsort(pp + 1, hi)
    #     qsort(lo, pp, depth + 1)

cdef inline void qsort_cmp(double *lo, double* hi):
    cdef:
        size_t n = hi - lo
        double* pp
        double* mid
        double p

    # base cases
    if n == 0:
        return
    if n == 1:
        if lo[0] > hi[0]:
            swap(lo, hi)
        return

    # insertsort on small casses
    if n < 16:
        isort(lo, hi)
        return

    p = med3(lo, hi)
    pp = part_h(lo, hi, p)

    qsort_cmp(lo, pp)
    qsort_cmp(pp + 1, hi)


cdef inline double* part_l(double *lo, double* hi, double* pp):
    """
    Lomuto's Partition Scheme. Keeps original pivot
    between two arrays.

    Partitions array around the pivot inplace:
    |  < p  | p |    > p    |
              ↑
            pivot

    Works when no duplicates.

    :param lo: starting pointer (included)
    :param hi: ending pointer (included)
    :param pp: pointer to pivot position
    :return: pointer the pivot in partitioned array
    """
    cdef:
        double* pi = lo + 1     # main iter ptr
        double* pj = lo + 1     # temporary pivot ptr

    swap(lo, pp) # swap pivot to first elem

    while pi <= hi:
        # if current elem smaller than pivot
        if pi[0] < lo[0]:
            swap(pi, pj)
            pj += 1  # inc temp pivot ptr
        pi += 1
    pj -= 1
    swap(lo, pj)
    return pj

cdef inline double* part_h(double *lo, double* hi, double p):
    """     
    Hoare Partition Scheme. Using double pointers moving
    towards each other. Splits [lo, hi] input array into two 
    parts: 
    (1) all elems <= p to the left side
    (2) elems >= p to the right
    returns pointer to the last elem from (1) 
    
    Partitions array around the pivot inplace:
    |  <= p  | |    >= p    |
              ↑
              j-th last elem of smaller array

    :param lo: starting pointer (included)
    :param hi: ending pointer (included)
    :param p: pivot value
    :return: pointer to last elem of smaller array

    """
    cdef:
        double* pi = lo - 1
        double* pj = hi + 1
    while True:
        pi += 1
        while pi[0] < p:
            pi += 1
        pj -= 1
        while pj[0] > p:
            pj -= 1
        if pi >= pj:
            return pj
        swap(pi, pj)


""" Python wrap """

def qsort_cy(arr):
    cdef:
        double* a
        size_t  n
    a, n = read_numpy(arr)
    qsort(a, a + n - 1)

def qsort_cmp_py(arr):
    cdef:
        double* lo
        double* hi
        size_t  n
    lo, n = read_numpy(arr)
    hi = lo + n - 1
    qsort_cmp(lo, hi)


def qsort_stack(arr):
    cdef:
        double*  lo
        double*  hi
        size_t   n, maxdepth
        stack_c* s

    lo, n = read_numpy(arr)
    hi = lo + n - 1
    maxdepth = 2 * log2(n)
    s = create_stack(2 * maxdepth)

    push(s, <size_t>lo)
    push(s, <size_t>hi)
    qsort_s(s)

    free_stack(s)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

def test_qsort():
    
    cdef:
        size_t n = 100
        double* a
        size_t i, j
        double [:] a_mv
    # np.random.seed(5)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        a_mv = np.sort(arr)
        qsort(a, a + n - 1)
        for j in range(n):
            assert a[j] == a_mv[j]


def test_qsort_cmp():
    
    cdef:
        size_t n = 100
        double* a
        size_t i, j
        double [:] np_sort
        double [:] q_sort

    # np.random.seed(5)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        np_sort = np.sort(arr)
        qsort_cmp_py(arr)
        q_sort = arr

        for j in range(n):
            if q_sort[j] != np_sort[j]:
                print(i)
                print(arr)
            assert q_sort[j] == np_sort[j]


def test_qsort_stack():
    
    cdef:
        size_t n = 100
        double* a
        size_t i, j
        double [:] np_sort
        double [:] q_sort

    # np.random.seed(5)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        np_sort = np.sort(arr)
        qsort_stack(arr)
        q_sort = arr

        for j in range(n):
            if q_sort[j] != np_sort[j]:
                print(i)
                print(arr)
            assert q_sort[j] == np_sort[j]


def test_part_l():
    
    cdef:
        size_t n = 100
        double* a
        double* pp
        double p
        size_t i, j, p_idx
        double [:] a_mv
    # np.random.seed(3)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        p_idx = frand() % n
        p = a[p_idx]

        # print(arr)
        pp = part_l(a, a + n - 1, &a[p_idx])
        # print(arr)
        for j in range(0, pp - a):
            assert a[j] <= p
        for j in range(pp - a, n - 1):
            assert a[j] >= p

def test_part_h():
    
    cdef:
        size_t n = 100
        double* a
        double* pp
        double p
        size_t i, j, p_idx
        double [:] a_mv
    # np.random.seed(3)
    for i in range(1000):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        p_idx = frand() % n
        p = a[p_idx]

        # print(arr)
        pp = part_h(a, a + n - 1, p)
        # print(arr)
        for j in range(0, pp - a + 1):
            assert a[j] <= p
        for j in range(pp - a + 1, n - 1):
            assert a[j] >= p