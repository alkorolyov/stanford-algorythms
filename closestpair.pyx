# cython: language_level  = 3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# distutils: extra_compile_args = /O2 /Ob3 /arch:AVX2 /openmp

cimport cython

cdef extern from "math.h":
    float sqrtf(float)
    
from libc.float cimport DBL_EPSILON
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport numpy as cnp

import numpy as np
from scipy.spatial.distance import pdist

cnp.import_array()

cdef struct arr_1d:
    double value
    size_t idx

cdef int cmp_arr(const void *a_ptr, const void *b_ptr):
    cdef double a = (<double *>a_ptr)[0]
    cdef double b = (<double *>b_ptr)[0]
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

cdef double distance(double x1, double y1, double x2, double y2):
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

cdef inline double dist_sqr(double x1, double y1, double x2, double y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

cdef double min_3_db(double a, double b, double c):
    if a < b and a < c:
        return a
    if b < a and b < c:
        return b
    return c

cdef double min_2_db(double a, double b):
    if a < b:
        return a
    else:
        return b

cdef size_t max_2_int(size_t a, size_t b):
    if a > b:
        return a
    else:
        return b

cdef size_t min_2_int(size_t a, size_t b):
    if a < b:
        return a
    else:
        return b


""" ######## Using memory views ############ """
cpdef double min_dist_naive_mv(double[:, :] arr):
    cdef size_t i, j, n
    cdef double min, dist
    n = arr.shape[0]
    min = dist_sqr(arr[0, 0], arr[0, 1],
                   arr[1, 0], arr[1, 1])
    for i in range(n - 1):
        for j in range(i + 1, n):
            # dist = dist_sqr(arr[i, 0], arr[i, 1], arr[j, 0], arr[j, 1])
            dist = (arr[i, 0] - arr[j, 0]) * (arr[i, 0] - arr[j, 0]) + (arr[i, 1] - arr[j, 1]) * (arr[i, 1] - arr[j, 1])
            if dist < min:
                min = dist
    return sqrt(min)


""" ########### Using direct memory access ########## """
cpdef double min_dist_naive(cnp.ndarray[double, ndim=2] arr):
    cdef cnp.npy_intp * dims
    cdef double * data
    if arr.flags['C_CONTIGUOUS']:
        dims = cnp.PyArray_DIMS(arr)
        data = <double *> cnp.PyArray_DATA(arr)
        return min_dist_naive_c(dims[0], data)
    else:
        print('Array is non C-contiguous')
        return -1


cdef double min_dist_naive_c(size_t n, double *arr):
    cdef size_t i, j
    cdef double min, dist
    min = dist_sqr(arr[0], arr[1],
                   arr[2], arr[3])
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = dist_sqr(arr[2*i], arr[2*i + 1], arr[2*j], arr[2*j + 1])
            # dist = (arr[2*i] - arr[2*j]) * (arr[2*i] - arr[2*j]) + (arr[2*i + 1] - arr[2*j + 1]) * (arr[2*i + 1] - arr[2*j + 1])
            if dist < min:
                min = dist
    return sqrt(min)

""" #################### Using numpy arrays memory views ##################### """

cpdef double min_dist_mv(cnp.ndarray[cnp.float64_t, ndim=2] P):
    idx = cnp.PyArray_ArgSort(P, 0, cnp.NPY_QUICKSORT)
    cdef double[:, :] Px = P[idx[:, 0]]
    cdef double[:, :] Py = P[idx[:, 1]]
    return sqrt(_min_dist(Px, Py))

cdef double _min_dist_split(double[:, :] Sy, double delta):
    cdef size_t i, j, n
    n = Sy.shape[0]
    for i in range(n - 1):
        for j in range((i + 1), min_2_int(i + 7, n)):
            delta = min_2_db(delta, dist_sqr(Sy[i, 0], Sy[i, 1], Sy[j, 0], Sy[j, 1]))
    return delta

cdef cnp.ndarray[cnp.float_t, ndim=2] create_array(size_t length):
    cdef cnp.npy_intp *l = [0, 2]
    l[0] = length
    cdef cnp.ndarray[cnp.float_t, ndim=2] buff = cnp.PyArray_SimpleNew(2, l, cnp.NPY_FLOAT64)
    return buff

cdef double _min_dist(double[:, :] Px, double[:, :] Py):
    cdef size_t i, j, k, n
    n = Px.shape[0]
    # base cases
    if n == 2:
        return dist_sqr(Px[0, 0], Px[0, 1], Px[1, 0], Px[1, 1])
    if n == 3:
        return min_3_db(dist_sqr(Px[0, 0], Px[0, 1], Px[1, 0], Px[1, 1]),
                            dist_sqr(Px[0, 0], Px[0, 1], Px[2, 0], Px[2, 1]),
                            dist_sqr(Px[1, 0], Px[1, 1], Px[2, 0], Px[2, 1]))

    cdef size_t mid = n // 2
    cdef double mid_x = Px[mid, 0]

    cdef double[:, :] Qx, Rx
    Qx, Rx = Px[:mid, :], Px[mid:, :]

    # copy Rx, Ry -> Qy, Ry
    cdef double[:, :] Qy = create_array(Qx.shape[0])
    cdef double[:, :] Ry = create_array(Rx.shape[0])

    j = 0
    k = 0
    # print("n=", n)
    # print("i  Py[i, 0], mid_x")
    for i in range(n):
        # print(i, Py[i, 0], "    ", mid_x)
        if Py[i, 0] < mid_x:
            Qy[j, 0] = Py[i, 0]
            Qy[j, 1] = Py[i, 1]
            j += 1
        else:
            Ry[k, 0] = Py[i, 0]
            Ry[k, 1] = Py[i, 1]
            k += 1

    # print("j, k")
    # print(j, k)
    # print("===========")

    cdef double d1 = _min_dist(Qx, Qy)
    cdef double d2 = _min_dist(Rx, Ry)
    delta = min_2_db(d1, d2)

    cdef double[:, :] Sy = create_array(n)
    j = 0
    for i in range(n):
        if (Py[i, 0] >= mid_x - sqrt(delta)) or (Py[i, 0] <= mid_x + sqrt(delta)):
            Sy[j, 0] = Py[i, 0]
            Sy[j, 1] = Py[i, 1]
            j += 1
    if j != 0:
        delta = _min_dist_split(Sy[:j, :], delta)
    return delta


"""################### Using arrays in pure C ######################"""
cpdef double min_dist_c(cnp.ndarray[cnp.float64_t, ndim=2] P, cnp.NPY_SORTKIND kind = cnp.NPY_QUICKSORT):
    cdef size_t i
    idx = cnp.PyArray_ArgSort(P, 0, kind)
    cdef double[:, :] Px_mview = P[idx[:, 0]]
    cdef double[:, :] Py_mview = P[idx[:, 1]]

    cdef double *Px
    cdef double *Py
    Px = <double *> malloc(Px_mview.shape[0] * 2 * sizeof(double))
    Py = <double *> malloc(Py_mview.shape[0] * 2 * sizeof(double))

    # # copy sorted arrays to Px, Py
    for i in range(Px_mview.shape[0]):
        Px[2*i] = Px_mview[i, 0]
        Px[2*i + 1] = Px_mview[i, 1]
        Py[2*i] = Py_mview[i, 0]
        Py[2*i + 1] = Py_mview[i, 1]

    cdef double d = _min_dist_c(Px_mview.shape[0], Px, Py)

    free(Px)
    free(Py)
    return sqrt(d)


cdef double _min_dist_c(size_t n, double *Px, double *Py):
    cdef size_t i, j, k
    # base cases
    if n == 2:
        return dist_sqr(Px[0], Px[1], Px[2*1], Px[2*1 + 1])
    if n == 3:
        return min_3_db(dist_sqr(Px[0], Px[1], Px[2], Px[2 + 1]),
                        dist_sqr(Px[0], Px[1], Px[2*2], Px[2*2 + 1]),
                        dist_sqr(Px[2], Px[2 + 1], Px[2*2], Px[2*2 + 1]))
    # split Px on Qx and Rx
    cdef size_t mid = n // 2
    cdef double mid_x = Px[2 * mid]
    cdef double *Qx
    cdef double *Rx
    Qx = Px                 # Px[:mid, :], total size 2 * mid
    Rx = Px + 2 * mid       # Px[mid:, :], total size 2 * (n - mid)

    cdef double *Qy
    cdef double *Ry
    Qy = <double*> malloc(mid * 2 * sizeof(double))
    Ry = <double*> malloc((n - mid) * 2 * sizeof(double))

    # sorting Qx, Rx by y, according to Py array
    _sort_y(n, mid_x, Qy, Ry, Py)

    cdef double d1 = _min_dist_c(mid, Qx, Qy)
    cdef double d2 = _min_dist_c(n - mid, Rx, Ry)
    cdef double delta = min_2_db(d1, d2)

    free(Qy)
    free(Ry)

    # get points in Sy
    cdef double *Sy
    Sy = <double*> malloc(n * 2 * sizeof(double))
    cdef size_t s_y_size = _get_sy(n, mid_x, delta, Py, Sy)
    delta = _min_dist_split_c(s_y_size, Sy, delta)
    free(Sy)
    return delta

cdef double _min_dist_split_c(size_t n, double *Sy, double delta):
    cdef size_t i, j, j_max
    cdef double d, d_max
    for i in range(n - 1):
    # for i in prange(n - 1,=True):
        # ~7% speed up compared to func call min_2_int()
        if i + 7 < n:
            j_max = i + 7
        else:
            j_max = n

        for j in range((i + 1), j_max):
            # ~30% speed up compared to func call dist_sqr()
            # d = dist_sqr(Sy[2*i], Sy[2*i + 1], Sy[2*j], Sy[2*j + 1])
            d = (Sy[2*i] - Sy[2*j]) * (Sy[2*i] - Sy[2*j]) + (Sy[2*i + 1] - Sy[2*j + 1]) * (Sy[2*i + 1] - Sy[2*j + 1])
            if d < delta:
                delta = d
    return delta

cdef void _sort_y(size_t n, double mid_x, double *Qy, double *Ry, double *Py):
    cdef size_t i, j, k
    # sorting Qx, Rx by y, according to Py array
    j = 0
    k = 0
    for i in range(n):
        if Py[2*i] < mid_x:
            Qy[2*j] = Py[2*i]
            Qy[2*j + 1] = Py[2*i + 1]
            j += 1
        elif Py[2*i] >= mid_x:
            Ry[2*k] = Py[2*i]
            Ry[2*k + 1] = Py[2*i + 1]
            k += 1
    return

cdef size_t _get_sy(size_t n, double mid_x, double delta, double *Py, double *Sy):
    cdef size_t i
    cdef size_t j = 0
    cdef double upper_bound = mid_x + sqrt(delta)
    cdef double lower_bound = mid_x - sqrt(delta)
    for i in range(n):
        if (Py[2*i] > lower_bound) or (Py[2*i] < upper_bound):
            Sy[2*j] = Py[2*i]
            Sy[2*j + 1] = Py[2*i + 1]
            j += 1
    return j


""" ################ 32 bit version ############################## """

cdef inline float dist32(float x1, float y1, float x2, float y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

cdef inline float min32(float a, float b):
    if a > b:
        return b
    else:
        return a
    
cdef float min_3(float a, float b, float c):
    if a < b and a < c:
        return a
    if b < a and b < c:
        return b
    return c


cpdef float min_dist32(cnp.ndarray[cnp.float32_t, ndim=2] P, cnp.NPY_SORTKIND kind = cnp.NPY_QUICKSORT):
    cdef size_t i
    idx = cnp.PyArray_ArgSort(P, 0, kind)
    cdef float[:, :] Px_mview = P[idx[:, 0]]
    cdef float[:, :] Py_mview = P[idx[:, 1]]

    cdef float *Px
    cdef float *Py
    Px = <float *> malloc(Px_mview.shape[0] * 2 * sizeof(float))
    Py = <float *> malloc(Py_mview.shape[0] * 2 * sizeof(float))

    # # copy sorted arrays to Px, Py
    for i in range(Px_mview.shape[0]):
        Px[2*i] = Px_mview[i, 0]
        Px[2*i + 1] = Px_mview[i, 1]
        Py[2*i] = Py_mview[i, 0]
        Py[2*i + 1] = Py_mview[i, 1]

    cdef float d = _mindist32(Px_mview.shape[0], Px, Py)
    free(Px)
    free(Py)
    return sqrtf(d)

cdef float _mindist32(size_t n, float *Px, float *Py):
    cdef size_t i, j, k
    # base cases
    if n == 2:
        return dist32(Px[0], Px[1], Px[2*1], Px[2*1 + 1])
    if n == 3:
        return min_3(dist32(Px[0], Px[1], Px[2], Px[2 + 1]),
                        dist32(Px[0], Px[1], Px[2*2], Px[2*2 + 1]),
                        dist32(Px[2], Px[2 + 1], Px[2*2], Px[2*2 + 1]))
    # split Px on Qx and Rx
    cdef size_t mid = n // 2
    cdef float mid_x = Px[2 * mid]
    cdef float *Qx
    cdef float *Rx
    Qx = Px                 # Px[:mid, :], total size 2 * mid
    Rx = Px + 2 * mid       # Px[mid:, :], total size 2 * (n - mid)

    cdef float *Qy
    cdef float *Ry
    Qy = <float*> malloc(mid * 2 * sizeof(float))
    Ry = <float*> malloc((n - mid) * 2 * sizeof(float))

    # sorting Qx, Rx by y, according to Py array
    _sort_y32(n, mid_x, Qy, Ry, Py)

    cdef float d1 = _mindist32(mid, Qx, Qy)
    cdef float d2 = _mindist32(n - mid, Rx, Ry)
    cdef float delta = min32(d1, d2)

    free(Qy)
    free(Ry)

    # get points in Sy
    cdef float *Sy
    Sy = <float*> malloc(n * 2 * sizeof(float))
    cdef size_t s_y_size = _get_sy32(n, mid_x, delta, Py, Sy)
    delta = _min_dist_split32(s_y_size, Sy, delta)
    free(Sy)
    return delta

cdef void _sort_y32(size_t n, float mid_x, float *Qy, float *Ry, float *Py):
    cdef size_t i,j,k
    # sorting Qx, Rx by y, according to Py array
    j = 0
    k = 0
    for i in range(n):
        if Py[2*i] < mid_x:
            Qy[2*j] = Py[2*i]
            Qy[2*j + 1] = Py[2*i + 1]
            j += 1
        elif Py[2*i] >= mid_x:
            Ry[2*k] = Py[2*i]
            Ry[2*k + 1] = Py[2*i + 1]
            k += 1
    return

cdef size_t _get_sy32(size_t n, float mid_x, float delta, float *Py, float *Sy):
    cdef size_t i
    cdef size_t j = 0
    cdef float upper_bound = mid_x + sqrtf(delta)
    cdef float lower_bound = mid_x - sqrtf(delta)
    for i in range(n):
        if (Py[2*i] > lower_bound) or (Py[2*i] < upper_bound):
            Sy[2*j] = Py[2*i]
            Sy[2*j + 1] = Py[2*i + 1]
            j += 1
    return j

cdef float _min_dist_split32(size_t n, float *Sy, float delta):
    cdef size_t i, j, j_max
    cdef float d
    for i in range(n - 1):
        # ~7% speed up compared to func call min_2_int()
        if i + 7 < n:
            j_max = i + 7
        else:
            j_max = n

        for j in range((i + 1), j_max):
            d = dist_sqr(Sy[2*i], Sy[2*j], Sy[2*i + 1], Sy[2*j + 1])
            # d = (Sy[2*i] - Sy[2*j]) * (Sy[2*i] - Sy[2*j]) + (Sy[2*i + 1] - Sy[2*j + 1]) * (Sy[2*i + 1] - Sy[2*j + 1])
            if d < delta:
                delta = d
    return delta

""" ################ RETURN MAX points in Sy search ###############"""
cdef size_t _get_sy_strict(size_t n, double mid_x, double delta, double *Py, double *Sy):
    cdef size_t i
    cdef size_t j = 0
    for i in range(n):
        if (Py[2*i] > mid_x - delta) or (Py[2*i] < mid_x + delta):
            Sy[2*j] = Py[2*i]
            Sy[2*j + 1] = Py[2*i + 1]
            j += 1
    return j



cpdef (size_t, double) max_points_c(cnp.ndarray[cnp.float64_t, ndim=2] P, bint strict = 0):
    idx = cnp.PyArray_ArgSort(P, 0, cnp.NPY_QUICKSORT)
    cdef double[:, :] Px = P[idx[:, 0]]
    cdef double[:, :] Py = P[idx[:, 1]]

    cdef double *Px_ptr
    cdef double *Py_ptr
    cdef size_t i
    Px_ptr = <double *> malloc(Px.shape[0] * 2 * sizeof(double))
    Py_ptr = <double *> malloc(Py.shape[0] * 2 * sizeof(double))

    for i in range(Px.shape[0]):
        Px_ptr[2*i] = Px[i, 0]
        Px_ptr[2*i + 1] = Px[i, 1]
        Py_ptr[2*i] = Py[i, 0]
        Py_ptr[2*i + 1] = Py[i, 1]

    cdef size_t mp = 0
    cdef double delta
    delta = _max_points(Px.shape[0], Px_ptr, Py_ptr, &mp, strict)
    free(Px_ptr)
    free(Py_ptr)
    return mp, delta

cdef double _max_points_split_c(size_t n, double *Sy, double delta, size_t *mp):
    cdef size_t i, j
    cdef double dist
    for i in range(n - 1):
        for j in range((i + 1), min_2_int(i + 7, n)):
            dist = distance(Sy[2*i], Sy[2*i + 1], Sy[2*j], Sy[2*j + 1])
            if dist < delta:
                delta = dist
                mp[0] = max_2_int(mp[0], j - i)
    return delta

cdef double _max_points(size_t n, double *Px, double *Py, size_t *mp, bint strict):
    cdef size_t i, j, k
    # base cases
    if n == 2:
        return distance(Px[0], Px[1], Px[2*1], Px[2*1 + 1])
    if n == 3:
        return min_3_db(distance(Px[0], Px[1], Px[2], Px[2 + 1]),
                            distance(Px[0], Px[1], Px[2*2], Px[2*2 + 1]),
                            distance(Px[2], Px[2 + 1], Px[2*2], Px[2*2 + 1]))

    # split Px on Qx and Rx
    cdef size_t mid = n // 2
    cdef double mid_x = Px[2 * mid]
    cdef double *Qx
    cdef double *Rx
    Qx = Px                 # Px[:mid, :], total size 2 * mid
    Rx = Px + 2 * mid       # Px[mid:, :], total size 2 * (n - mid)

    cdef double *Qy
    cdef double *Ry
    Qy = <double*> malloc(mid * 2 * sizeof(double))
    Ry = <double*> malloc((n - mid) * 2 * sizeof(double))

    # sorting Qx, Rx by y, according to Py array
    _sort_y(n, mid_x, Qy, Ry, Py)

    cdef double d1
    cdef size_t mp1 = 0
    cdef double d2
    cdef size_t mp2 = 0

    d1 = _max_points(mid, Qx, Qy, &mp1, strict)
    d2 = _max_points(n - mid, Rx, Ry, &mp2, strict)
    cdef double delta = min_2_db(d1, d2)

    mp[0] = max_2_int(mp1, mp2)

    free(Qy)
    free(Ry)

    # get points in Sy
    cdef double *Sy
    Sy = <double*> malloc(n * 2 * sizeof(double))
    cdef size_t s_y_size
    if strict:
        s_y_size = _get_sy_strict(n, mid_x, delta, Py, Sy)
    else:
        s_y_size = _get_sy(n, mid_x, delta, Py, Sy)
    if s_y_size != 0:
        delta = _max_points_split_c(s_y_size, Sy, delta, mp)
    free(Sy)
    return delta

""" #############################################################
    ###################### UNIT TESTS ###########################
    ############################################################# 
"""

def test_min_dist_naive_c_1():
    cdef double *a = [0.0, 0.0, 4.0, 3.0, 9.0, 7.0]
    assert fabs(min_dist_naive_c(3, a) - 5.0) < 1e-16

def test_min_dist_c_1():
    cdef double *a = [0.0, 0.0, 4.0, 3.0, 9.0, 7.0]
    assert fabs(_min_dist_c(3, a, a) - 25.0) < 1e-16

def test_min_dist_c_2():
    cdef double *px = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6]
    cdef double *py = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6]
    # arr = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.5, 0.6]])

    assert fabs(sqrt(_min_dist_c(4, px, px)) - min_dist_naive_c(4, px)) < 1e-16

def test_min_dist_c_3():
    for i in range(100):
        arr = np.random.randn(20, 2)
    assert fabs(min_dist_c(arr) - pdist(arr).min()) < 1e-16


def test_mindist32_1():
    cdef float *x32 = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6]
    cdef float *y32 = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6]
    cdef double *x64 = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6]
    cdef double *y64 = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6]
    # arr = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.5, 0.6]])

    assert fabs(<double>_mindist32(4, x32, y32) - _min_dist_c(4, x64, y64)) < 1e-8
    assert fabs(<double>sqrt(_mindist32(4, x32, y32)) - min_dist_naive_c(4, x64)) < 1e-8
