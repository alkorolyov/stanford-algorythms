from libc.stdlib cimport RAND_MAX
from c_utils cimport frand, frand32, read_numpy, log2, print_arr
from sorting cimport _swap, choose_p
from insertsort cimport insertsort
from heapsort cimport hsort_c
from stack cimport stack_c, create_stack, push, pop, size_s, is_empty_s, free_stack


from utils import print_func_name
import numpy as np

cdef void qsort_s(stack_c* stack):
    """ QuickSort stack version """
    cdef:
        double* arr
        size_t i, j, n, maxdepth
        size_t p_idx, idx, delta

    maxdepth = pop(stack)

    while not is_empty_s(stack):
        n = pop(stack)
        arr = <double*>pop(stack)

        # base cases
        if n <= 1:
            continue
        if n == 2:
            if arr[0] > arr[1]:
                _swap(arr, 0, 1)
            continue

        # insertsort on small casses
        if n <= 16:
            insertsort(arr, n)
            continue

        # choose pivot
        if n > RAND_MAX:
            p_idx = frand32() % n
        else:
            p_idx = frand() % n

        idx = partition_l(arr, n, p_idx)
        delta = idx + 1

        # qsort(arr, idx)
        push(stack, <size_t>arr)
        push(stack, idx)
        # qsort(arr + delta, n - delta)
        push(stack, <size_t>(arr + delta))
        push(stack, n - delta)

""" ================= QuickSort in C ================== """

cdef void qsort(double *arr, size_t n):
    cdef:
        double x
        size_t i, j, p_idx, idx, delta

    # base cases
    if n <= 1:
        return
    if n == 2:
        if arr[0] > arr[1]:
            _swap(arr, 0, 1)
        return

    # insertsort on small casses
    if n <= 16:
        insertsort(arr, n)
        return

    """ different choose pivot options """

    # if n > RAND_MAX:
    #     p_idx = frand32() % n
    # else:
    #     p_idx = frand() % n

    # p_idx = frand32() % n
    # p_idx = frand() % n
    # p_idx = rand() % n
    # p_idx = 0 # first
    # p_idx = n - 1 # last
    p_idx = n // 2 # middle
    # p_idx = choose_p(arr, n) # median of 3

    # Lomuto partition
    idx = partition_l(arr, n, p_idx)
    delta = idx + 1

    if idx < n - delta:
        qsort(arr, idx)
        qsort(arr + delta, n - delta)
    else:
        qsort(arr + delta, n - delta)
        qsort(arr, idx)

    # # Hoare partition
    # idx = partition_h(arr, n, p_idx)
    # delta = idx + 1
    #
    # if delta < n - delta:
    #     qsort(arr, delta)
    #     qsort(arr + delta, n - delta)
    # else:
    #     qsort(arr + delta, n - delta)
    #     qsort(arr, delta)


cdef inline size_t partition_l(double *arr, size_t n, size_t p_idx):
    """
    Lomuto's Partition Scheme. Keeps original pivot
    between two arrays. 
    
    Partitions array around the pivot inplace:
    |  < p  | p |    > p    |
              ↑
            pivot

    Works when no duplicates.

    :param arr: input array
    :param n: array length
    :param p_idx: pivot index in the input array
    :return: index of pivot in partitioned array
    """
    cdef size_t i
    cdef size_t j = 1
    _swap(arr, 0, p_idx)
    for i in range(1, n):
        if arr[i] < arr[0]:
            _swap(arr, i, j)
            j += 1
    j -= 1
    _swap(arr, 0, j)
    return j

cdef inline partition_h(double *arr, size_t n, size_t p_idx):
    """ 
    Hoare Partition Scheme. Using double pointers moving
    towards each other. Splits into two arrays - smaller or
    equal than pivot to the left side, bigger or equal - to
    the right.
    
    Partitions array around the pivot inplace:
    |    <= p  |   |  >= p  |
                 ↑
                j-th last elem of smaller array

    """
    cdef:
        size_t i, j
        double p = arr[p_idx]
    i = -1
    j = n
    while True:
        i += 1
        while arr[i] < p: # first el a[i] >= p
            i += 1
        j -= 1
        while arr[j] > p: # stop when a[j] <= p
            j -= 1

        if i >= j:
            return j
        _swap(arr, i, j)

def qsort_py(arr):
    cdef:
        double*     a
        size_t      n

    a, n = read_numpy(arr)
    qsort(a, n)

def qsort_stack(arr):
    cdef:
        double*     a
        size_t      n, maxdepth
        stack_c*    s


    a, n = read_numpy(arr)
    maxdepth = 2 * log2(n)
    s = create_stack(2 * maxdepth)

    push(s, <size_t>a)
    push(s, n)
    push(s, maxdepth)

    qsort_s(s)

    free_stack(s)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_qsort():
    print_func_name()
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
        qsort(a, n)
        for j in range(n):
            assert a[j] == a_mv[j]

def test_qsort_stack():
    print_func_name()
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

def test_partition_hoare():
    print_func_name()
    cdef:
        size_t n = 5
        double* a
        double p
        size_t i, j, p_idx
        double [:] a_mv
    np.random.seed(3)
    for i in range(10):
        arr = np.random.randint(0, n, n).astype(np.float64)
        a, n = read_numpy(arr)
        p_idx = frand() % n
        p_idx = n // 2
        # p_idx = 0
        p = a[p_idx]
        p_idx = partition_h(a, n, p_idx)

        for j in range(p_idx + 1):
            assert a[j] <= p
        for j in range(p_idx + 1, n):
            assert a[j] >= p


