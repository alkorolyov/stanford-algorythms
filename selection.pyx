# cython: language_level=3
# cython: profile=False
# cython: linetrace=False
# cython: binding=False


# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# distutils: extra_compile_args = /O2 /Ob3 /arch:AVX2

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport rand, srand
from libc.time cimport time
from sorting cimport partition_c, partition3_c, qsort_c, msort_c, choose_p
from utils import print_func_name
cimport numpy as np
np.import_array()


cdef inline size_t div(size_t n, size_t k):
    if n % k > 0:
        return n // k + 1
    else:
        return n // k

cdef inline double _min(double a, double b):
    if a < b:
        return a
    else:
        return b

cdef inline double _max(double a, double b):
    if a > b:
        return a
    else:
        return b

cdef inline void _swap(double *a, size_t i, size_t j):
    cdef double t = a[i]
    a[i] = a[j]
    a[j] = t

cdef inline void _sort(double *a, size_t i, size_t j):
    if a[i] > a[j]:
        _swap(a, i, j)

cdef inline double median_c(double *arr, size_t n):
    qsort_c(arr, n)
    return arr[n // 2]

cdef inline double median5(double *a):
    _sort(a, 0, 1)
    _sort(a, 3, 4)
    if a[0] > a[3]:
        _swap(a, 0, 3)
        _swap(a, 1, 4)

    if a[2] > a[1]:
        if a[1] < a[3]:
            return _min(a[2], a[3])
        else:
            return _min(a[1], a[4])

    # if a[2] <= a[1]:
    else:
        if a[2] > a[3]:
            return _min(a[2], a[4])
        else:
            return _min(a[1], a[3])

cpdef double r_select(np.ndarray[double, ndim=1] arr, size_t k):
    cdef np.npy_intp * dims
    cdef double * data
    if arr.flags['C_CONTIGUOUS']:
        dims = np.PyArray_DIMS(arr)
        data = <double *> np.PyArray_DATA(arr)
        return r_select_c(data, dims[0], k)
    else:
        print('Array is non C-contiguous')

cdef double r_select_c(double *arr, size_t n, size_t k):

    # print(f" ======== n: {n}, k: {k} =======", )
    # for i in range(n):
    #     print(f"a[{i}]", arr[i])

    if n == 1:
        return arr[0]

    cdef size_t p_idx = choose_p(arr, n)
    # cdef size_t p_idx = rand() % n
    cdef size_t idx = partition_c(arr, n, p_idx)
    # print("pivot:", arr[idx], "idx:", idx)
    # for i in range(n):
    #     print(f"a[{i}]", arr[i])

    if idx == k - 1:
        return arr[idx]
    elif k - 1 < idx:
        return r_select_c(arr, idx, k)
    elif k - 1 > idx:
        return r_select_c(arr + idx, n - idx, k - idx)

cpdef double d_select(np.ndarray[double, ndim=1] arr, size_t k):
    cdef np.npy_intp *dims
    cdef double *data
    cdef double *buff
    cdef double res
    if arr.flags['C_CONTIGUOUS']:
        dims = np.PyArray_DIMS(arr)
        data = <double *> np.PyArray_DATA(arr)
        buff = <double *> PyMem_Malloc(dims[0] // 3 * sizeof(double)) # 1/5 + 1/25 + 1/125 + ... = 1/4
        res = d_select_c(data, dims[0], k, buff)
        PyMem_Free(buff)
        return res
    else:
        print('Array is non C-contiguous')

cdef double d_select_c(double *arr, size_t n, size_t k, double *buff, bint pivot_call = False):
    """
    "Median of medians" approach for selection of k-th order statistics
    :param arr: input array
    :param n: array length
    :param k: int order statistics
    :param buff: pointer for array of medians
    :param pivot_call: bool if recursive call for pivot or main loop
    :return: value of k-th order statistics
    """
    # cdef double *m_arr = <double*>PyMem_Malloc(n * sizeof(double))
    cdef double *m_arr = buff
    cdef size_t m_len, i

    # if pivot_call:
    #     print(" ===== pivot search ======")
    # else:
    #     print("===== MAIN LOOP =====")
    #
    # print("n:", n, "k:", k)

    if n == 1:
        # print("return: ", arr[0])
        return arr[0]

    # print("input array ")
    # for i in range(n):
    #     print(f"a[{i}]", arr[i])

    # creat array of medians
    m_len = div(n, 5)

    cdef size_t l
    for i in range(m_len):
        l = n - i * 5
        if l >= 5:
            # m_arr[i] = median_c(arr + i*5, 5)
            m_arr[i] = median5(arr + i*5)
        elif l == 1:
            m_arr[i] = arr[i*5]
        else:
            m_arr[i] = median_c(arr + i*5, l)

    # print("m_len", m_len)
    # for i in range(m_len):
    #     print(f"m_arr[{i}]", m_arr[i])

    # recursively find the pivot (median)
    cdef double p
    if n < 10:
        # print("## n < 10 ##")
        p = d_select_c(m_arr, m_len, 1, buff + m_len, True)
    else:
        p = d_select_c(m_arr, m_len, n // 10, buff + m_len, True)

    # if not pivot_call:
    #     print(" ==== pivot found ==== ")

    # PyMem_Free(m_arr)


    # print("p", p)
    # find the pivot index in arr
    for i in range(n):
        if p == arr[i]:
            p_idx = i

    # print("p_idx", p_idx, "/", n)


    # partition array around the pivot
    cdef size_t idx = partition3_c(arr, n, p_idx)

    # print("idx", idx, "/", n)
    # print("partitioned array: ")
    # for i in range(n):
    #     print(f"a[{i}]", arr[i])


    if idx == k - 1:
        return arr[idx]
    elif idx > k - 1:
        return d_select_c(arr, idx, k, buff)
    elif idx < k - 1:
        return d_select_c(arr + idx, n - idx, k - idx, buff)


""" #############################################################
    ###################### UNIT TESTS ###########################
    ############################################################# 
"""

def test_r_select_c_1():
    print_func_name()
    cdef double *arr = [0.3, 0.2, 0.1]
    cdef size_t n = 3
    cdef size_t k = 1
    cdef size_t i
    for i in range(100):
        srand(time(NULL))
        assert r_select_c(arr, n, k) == 0.1

def test_r_select_c_2():
    print_func_name()
    cdef double *arr = [0.3, 0.2, 0.1]
    cdef size_t n = 3
    cdef size_t k = 2
    cdef size_t i
    for i in range(100):
        srand(time(NULL))
        assert r_select_c(arr, n, k) == 0.2

def test_r_select_c_3():
    print_func_name()
    cdef double *arr = [0.3, 0.2, 0.1]
    cdef size_t n = 3
    cdef size_t k = 3
    cdef size_t i
    for i in range(100):
        srand(time(NULL))
        assert r_select_c(arr, n, k) == 0.3

def test_r_select_c_4():
    print_func_name()
    cdef:
        double arr[20]
        size_t n = 20
        double q, r
        size_t i, j, k
    for i in range(1000):
        srand(time(NULL))
        k = rand() % (n - 1) + 1
        for j in range(n):
            arr[j] = rand()
        r = r_select_c(arr, n, k)
        qsort_c(arr, n)
        q = arr[k - 1]
        assert q == r

def test_median5_1():
    print_func_name()
    cdef double *arr = [0.3, 0.2, 0.1, 0, -0.1]
    assert median5(arr) == 0.1

def test_median5_2():
    print_func_name()
    cdef double *arr = [0.3, 0.3, 0.2, 0.1, 0.1]
    assert median5(arr) == 0.2

def test_median5_3():
    print_func_name()
    cdef double *arr = [0.3, 0.3, 0.2, 0.2, 0.1]
    assert median5(arr) == 0.2

def test_median5_4():
    print_func_name()
    cdef:
        double arr[5]
        size_t n = 5
        double q, r
        size_t i, j

    srand(time(NULL))
    for i in range(1000):
        for j in range(n):
            arr[j] = rand()
        r = median5(arr)
        qsort_c(arr, n)
        q = arr[2]
        assert q == r


def test_median_c_1():
    print_func_name()
    cdef:
        double *arr = [0.3, 0.2, 0.1, 0, -0.1]
        size_t n = 5
    assert median_c(arr, n) == 0.1

def test_median_c_2():
    print_func_name()
    cdef:
        double *arr = [0.3, 0.2, 0.1, 0]
        size_t n = 4
    assert median_c(arr, n) == 0.2

def test_median_c_3():
    print_func_name()
    cdef:
        double *arr = [0.3, 0.2]
        size_t n = 2
    assert median_c(arr, n) == 0.3

def test_median_c_4():
    print_func_name()
    cdef:
        double *arr = [0.3]
        size_t n = 1
    assert median_c(arr, n) == 0.3

def test_d_select_1():
    print_func_name()
    cdef:
        double *arr = [0.1 , 0.2]
        double buff[2]
        size_t n = 2
        size_t k = 2
    assert d_select_c(arr, n, k, buff) == 0.2

def test_d_select_2():
    print_func_name()
    cdef:
        double *arr = [0.1 , 0.2]
        double buff[2]
        size_t n = 2
        size_t k = 1
    assert d_select_c(arr, n, k, buff) == 0.1

def test_d_select_3():
    print_func_name()
    cdef:
        double *arr = [1.1, 1.0 , 0.9 , 0.8 , 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        double buff[11]
        size_t n = 11
        size_t k = 4
    assert d_select_c(arr, n, k, buff) == 0.4

def test_d_select_4():
    print_func_name()
    cdef:
        double *arr = [0.1, 0.1 , 0.2]
        double buff[3]
        size_t n = 3
        size_t k = 2
    assert d_select_c(arr, n, k, buff) == 0.1

def test_d_select_5():
    print_func_name()
    cdef:
        double *arr = [0.4, 0.3, 0.1, 0.1]
        double buff[3]
        size_t n = 4
        size_t k = 1
    srand(time(NULL))
    assert d_select_c(arr, n, k, buff) == 0.1


def test_d_select_6():
    print_func_name()
    cdef:
        double arr[100]
        double buff[100]
        size_t n = 100
        double q, r
        size_t i, j, k

    srand(time(NULL))
    for i in range(1000):
        k = rand() % (n - 1) + 1
        # k = n - 1
        for j in range(n):
            arr[j] = rand()
        r = d_select_c(arr, n, k, buff)
        qsort_c(arr, n)
        q = arr[k - 1]
        assert q == r