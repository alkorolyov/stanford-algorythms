

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport rand
from c_utils cimport read_numpy, fastrand
cimport numpy as np
np.import_array()

# python wrap
cpdef void quicksort_c(np.ndarray[double, ndim=1] arr):
    """
    :param arr: C-contiguous numpy 1D array of doubles   
    """
    cdef:
        double* data
        size_t  size
    data, size = read_numpy(arr)
    qsort_c(data, size)

cpdef void quicksort_mv(np.ndarray[double, ndim=1] arr):
    qsort(arr)
    return

""" ################# QuickSort: C array ################### """

cdef inline void _swap(double *a, size_t i, size_t j):
    cdef double t = a[i]
    a[i] = a[j]
    a[j] = t

cdef inline void _sort(double *a, size_t i, size_t j):
    if a[i] > a[j]:
        _swap(a, i, j)

cdef inline size_t choose_p(double *arr, size_t n):
    """ 
    Choose pivot by median of 3 approach:
    median of first, middle and last elements
    :param arr: input array 
    :param n: array size
    :return: pivot index

    """
    cdef double a[3]
    a[0] = arr[0]
    a[1] = arr[n // 2]
    a[2] = arr[n - 1]

    if a[0] < a[1]:
        # a[0] < a[1]
        if a[2] > a[1]:
            # median is a[1]
            return n // 2
        elif a[2] < a[0]:
            # median is a[0]
            return 0
        else:
            # median is a[2]
            return n - 1
    else:
        # a[1] <= a[0]
        if a[2] > a[0]:
            # median is a[0]
            return 0
        elif a[2] < a[1]:
            # median is a[1]
            return n // 2
        else:
            # median is a[2]
            return n - 1

cdef void qsort_c(double *arr, size_t n):
    if n <= 1:
        return
    if n == 2:
        if arr[0] > arr[1]:
            _swap(arr, 0, 1)
        return

    """ different choose pivot options """
    cdef size_t p_idx
    p_idx = fastrand() % n
    # p_idx = rand() % n
    # p_idx = 0 # first
    # p_idx = n - 1 # last
    # p_idx = choose_p(arr, n) # median of 3

    cdef size_t idx = partition_c(arr, n, p_idx)
    cdef size_t delta = idx + 1
    qsort_c(arr, idx)
    qsort_c(arr + delta, n - delta)


cdef inline size_t partition_c(double *arr, size_t n, size_t p_idx):
    """
    Partitions array around the pivot inplace:
    |  < p  | p |    > p    |
              |
            pivot

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
    _swap(arr, 0, j - 1)
    return j - 1

cdef size_t partition3_c(double *arr, size_t n, size_t p_idx):
    """
    Partitions array in three parts from left to right:
    
    |  < p  |  equal p | p |  > p  |
                         |
                       pivot
    
    Used in "median of medians"
         
    :param arr: input array
    :param n: array length
    :param p_idx: pivot index in the input array
    :return: index of pivot in partitioned array
    """
    cdef size_t i, k
    cdef size_t j = 0

    _swap(arr, n - 1, p_idx)

    # for i in range(n):
    #     print(f"a[{i}]", arr[i])
    # print("======")

    for i in range(n - 1):
        if arr[i] < arr[n - 1]:
            _swap(arr, i, j)
            j += 1

    # for i in range(n):
    #     print(f"a[{i}]", arr[i])
    # print("======")
    # print("j", j)

    k = j
    for i in range(j, n - 1):
        if arr[i] == arr[n - 1]:
            _swap(arr, i, k)
            k += 1

    # for i in range(n):
    #     print(f"a[{i}]", arr[i])
    # print("======")

    # print("k", k)

    _swap(arr, n - 1, k)

    # for i in range(n):
    #     print(f"a[{i}]", arr[i])
    # print("======")

    return k



""" ################# QuickSort: Memory slice ################### """
cdef size_t randint(size_t lower, size_t upper):
    return rand() % (upper - lower + 1) + lower

cdef void swap(double [:] arr, size_t i, size_t j):
    cdef double tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


cdef void qsort(double [:] arr):
    if arr.shape[0] <= 1:
        return
    cdef size_t idx = partition(arr)
    cdef double [:] left = arr[:idx]
    cdef double [:] right = arr[idx + 1:]
    qsort(left)
    qsort(right)


cdef size_t partition(double [:] arr):
    cdef size_t i
    cdef size_t j = 1
    cdef double tmp
    # cdef size_t p_idx = randint(0, arr.shape[0] - 1)
    cdef size_t p_idx = rand() % arr.shape[0]
    # swap(arr, 0, p_idx)
    tmp = arr[0]
    arr[0] = arr[p_idx]
    arr[p_idx] = tmp

    for i in range(1, arr.shape[0]):
        if arr[i] < arr[0]:
            # swap(arr, i, j)
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            j += 1
    # swap(arr, 0, j - 1)
    tmp = arr[0]
    arr[0] = arr[j - 1]
    arr[j - 1] = tmp
    return j - 1

""" ############### MergeSort C ##################### """
cpdef void mergesort_c(np.ndarray[double, ndim=1] arr):
    """
    :param arr: c-contiguous numpy 1D array of doubles   
    """
    cdef:
        size_t  size
        double* data
        double* buff

    data, size = read_numpy(arr)
    buff = <double *> PyMem_Malloc(size * sizeof(double))
    msort_c(data, size, buff)
    PyMem_Free(buff)

cdef void msort_c(double *arr, size_t n, double *buff):
    # base case
    if n == 1:
        return

    # split
    cdef size_t idx = n // 2
    cdef double *a = arr
    cdef double *b = arr + idx

    # debug
    # print(f"=== n: {n} idx: {idx} ===", )
    # for i in range(idx):
    #     print(f"a[{i}]", a[i])
    #
    # for i in range(idx, n):
    #     print(f"b[{i}]", a[i])

    msort_c(a, idx, buff)
    msort_c(b, n - idx, buff)
    merge_c(a, b, n, idx, buff)

cdef void merge_c(double *a, double *b, size_t n, size_t idx, double *c):
    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k

    # print("==================================================")
    for k in range(n):
        # debug
        # print("k", k)
        # print(f"    a[{i}]", a[i])
        # print(f"    b[{j}]", b[j])

        # if last "a" element added - continue with "b" only
        if i == idx:
            c[k] = b[j]
            j += 1
            # print("        c[k]", c[k])
            continue
        # or finished iter through "b" continue with "a" only
        elif j == n - idx:
            c[k] = a[i]
            i += 1
            # print("        c[k]", c[k])
            continue

        # normal run
        if a[i] < b[j]:
            c[k] = a[i]
            i += 1
        else:
            c[k] = b[j]
            j += 1
        # print("        c[k]", c[k])

    for i in range(n):
        a[i] = c[i]


""" ############### HeapSort C ##################### """


""" #############################################################
    ###################### UNIT TESTS ###########################
    ############################################################# 
"""

def test_swap_c():
    cdef double *arr = [0.1, -0.1]
    cdef size_t i = 0
    cdef size_t j = 1

    _swap(arr, i, j)

    assert arr[0] == -0.1
    assert arr[1] == 0.1

def test_partition_c_1():
    cdef double *arr = [0.4, 0.3, 0.2, 0.1]
    cdef size_t p_idx = 1
    cdef size_t n = 4
    assert partition_c(arr, n, p_idx) == 2
    assert arr[0] == 0.1
    assert arr[1] == 0.2
    assert arr[2] == 0.3
    assert arr[3] == 0.4

def test_partition3_c_1():
    cdef double *arr = [0.2, 0.3, 0.2, 0.1, 0.2]
    cdef size_t p_idx = 2
    cdef size_t n = 5
    assert partition3_c(arr, n, p_idx) == 3
    assert arr[0] == 0.1
    assert arr[1] == 0.2
    assert arr[2] == 0.2
    assert arr[3] == 0.2
    assert arr[4] == 0.3


def test_qsort_c_1():
    cdef double *arr = [0.3, 0.2, 0.1]
    cdef size_t n = 3
    qsort_c(arr, n)
    assert arr[0] == 0.1
    assert arr[1] == 0.2
    assert arr[2] == 0.3

def test_qsort_c_2():
    cdef double *arr = [0.2, 0.2, 0.1]
    cdef size_t n = 3
    qsort_c(arr, n)
    assert arr[0] == 0.1
    assert arr[1] == 0.2
    assert arr[2] == 0.2

def test_merge_c_1():
    cdef double *a = [0.1, 0.3]
    cdef double *b = [0.2, 0.4]
    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 4, 2, c)
    PyMem_Free(c)
    assert a[0] == 0.1
    assert a[1] == 0.2
    assert a[2] == 0.3
    assert a[3] == 0.4

def test_merge_c_2():
    cdef double *a = [0.2, 0.4]
    cdef double *b = [0.1, 0.3]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 4, 2, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.2
    assert a[2] == 0.3
    assert a[3] == 0.4

def test_merge_c_3():
    cdef double *a = [0.2, 0.3]
    cdef double *b = [0.1]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 3, 2, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.2
    assert a[2] == 0.3

def test_merge_c_4():
    cdef double *a = [0.1]
    cdef double *b = [0.2, 0.3]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 3, 1, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.2
    assert a[2] == 0.3

def test_merge_c_5():
    cdef double *a = [0.1, 0.1]
    cdef double *b = [0.2, 0.3]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 4, 2, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.1
    assert a[2] == 0.2
    assert a[3] == 0.3

def test_merge_c_6():
    cdef double *a = [0.1, 0.1]
    cdef double *b = [0.2]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 3, 2, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.1
    assert a[2] == 0.2

def test_merge_c_7():
    cdef double *a = [0.2]
    cdef double *b = [0.1, 0.1]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 3, 1, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.1
    assert a[2] == 0.2

def test_merge_c_8():
    cdef double *a = [0.1]
    cdef double *b = [0.1]

    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    merge_c(a, b, 2, 1, c)
    # for i in range(2):
    #     print(f"a[{i}]", a[i])
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.1


def test_msort_c_1():
    cdef double *a = [0.3, 0.2, 0.1]
    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    msort_c(a, 3, c)
    PyMem_Free(c)

    assert a[0] == 0.1
    assert a[1] == 0.2
    assert a[2] == 0.3

def test_msort_c_2():
    cdef double *a = [0.3, 0.2, 0.2]
    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    msort_c(a, 3, c)
    PyMem_Free(c)

    assert a[0] == 0.2
    assert a[1] == 0.2
    assert a[2] == 0.3

def test_msort_c_3():
    cdef double *a = [0.2, 0.3, 0.2]
    cdef double *c = <double *> PyMem_Malloc(10 * sizeof(double))
    msort_c(a, 3, c)
    PyMem_Free(c)

    assert a[0] == 0.2
    assert a[1] == 0.2
    assert a[2] == 0.3

