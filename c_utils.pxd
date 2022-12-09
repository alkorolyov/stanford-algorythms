
cimport numpy as np
from numpy cimport npy_intp
from cpython.object cimport PyObject

cdef:
    extern from "emmintrin.h":
        ctypedef double __m128d

        # min/max
        __m128d _mm_max_pd (__m128d a, __m128d b)
        __m128d _mm_min_pd(__m128d a, __m128d b)

        double _mm_cvtsd_f64(__m128d a) # movsd m64, xmm

        __m128d _mm_set_pd(double e1, double e0)

        # load
        __m128d _mm_loadu_pd(double* mem_addr) # movupd xmm, m128
        __m128d _mm_load_sd(double* mem_addr)  # movsd xmm, m64

        # store
        void _mm_storeu_pd(double * mem_addr, __m128d a) # movupd m128, xmm
        void _mm_store_sd(double * mem_addr, __m128d a) # movsd m64, xmm

        # 16-bit aligned needed
        void _mm_store_pd(double * mem_addr, __m128d a) # movapd m128, xmm
        __m128d _mm_load_pd (double* mem_addr) #movapd xmm, m128

    extern from "immintrin.h":
        ctypedef double __m256d
        __m256d _mm256_max_pd (__m256d a, __m256d b)


cdef class PyArrayObject:
    cdef:
        char *data
        int nd
        npy_intp *dimensions
        npy_intp *strides
        PyObject *base

cdef inline (double*, size_t) read_numpy(np.ndarray arr):
    return <double*>np.PyArray_DATA(arr), <size_t>np.PyArray_DIMS(arr)[0]


ctypedef unsigned short uint16_t
ctypedef unsigned long uint32_t
ctypedef unsigned long long uint64_t

cdef extern int _rdrand64_step(size_t*)
cdef extern int _rdrand32_step(size_t*)
cdef extern int _rdrand16_step(size_t*)


cdef size_t seed

cdef inline void sfrand(size_t s):
    global seed
    seed = s

cdef inline size_t frand():
    """
    Compute a pseudorandom integer.
    Output value in range [0, 32767]
    """
    global seed
    seed = (214013 * seed + 2531011)
    return (seed>>16)&0x7FFF


cdef:
    size_t rng_state
    size_t rng_inc

cdef inline uint32_t frand32():
    """ 
        PSG32 pseudorandom generator.
        Output value in range [0, 2^32 - 1]
    """
    global rng_state
    global rng_inc

    cdef size_t oldstate = rng_state
    # Advance internal state
    rng_state = oldstate * 6364136223846793005ULL + (rng_inc | 1)
    # Calculate output function (XSH RR), uses old state for max ILP
    cdef:
        size_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        size_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))


ctypedef struct pcg32_t:
    uint64_t state
    uint64_t inc

cdef inline void srand32(pcg32_t* rng, uint64_t initstate=0, uint64_t initseq=0):
    rng.state = 0U
    rng.inc = (initseq << 1u) | 1u
    frand32_loc(rng)
    rng.state += initstate
    frand32_loc(rng)


cdef inline uint32_t frand32_loc(pcg32_t* rng):
    """ 
        Compute a pseudorandom integer.
        Output value in range [0, 2^32 - 1]
    """
    cdef uint64_t oldstate = rng.state;
    # Advance internal state
    rng.state = oldstate * 6364136223846793005ULL + (rng.inc|1)
    # Calculate output function (XSH RR), uses old state for max ILP
    cdef:
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

cdef extern size_t _lzcnt_u64 (size_t x)

cdef inline size_t log2(size_t x):
    if x == 0:
        return 0
    return 63 - _lzcnt_u64(x)


cdef inline void swap(double* a, double* b) nogil:
    cdef double tmp = a[0]
    a[0] = b[0]
    b[0] = tmp


cdef inline void swap_s(size_t* a, size_t* b) nogil:
    cdef size_t tmp = a[0]
    a[0] = b[0]
    b[0] = tmp


cdef inline (double, double) med3_sse(double* lo1, double* hi2, double* hi1):
    cdef:
        __m128d l
        __m128d h
        __m128d m
        __m128d t1
        __m128d t2
        __m128d t3
        double* lo2 = hi1 + 1
        double* mid1 = lo1 + ((hi1 - lo1) >> 1)
        double* mid2 = lo2 + ((hi2 - lo2) >> 1)
        double p[2]

    l = _mm_set_pd(lo1[0], lo2[0])
    m = _mm_set_pd(mid1[0], mid2[0])
    h = _mm_set_pd(hi1[0], hi2[0])

    t1 = _mm_min_pd(l, m)
    t2 = _mm_max_pd(l, m)
    t3 = _mm_min_pd(h, t2)

    t2 = _mm_max_pd(t1, t3)

    _mm_store_pd(<double*>&p, t2)

    return p[0], p[1]


cdef inline double imed3(double* lo, double* hi):
    cdef:
        __m128d l
        __m128d h
        __m128d m
        __m128d t1
        __m128d t2
        __m128d t3
        double* mid = lo + ((hi - lo) >> 1)
        double p

    l = _mm_load_sd(lo)
    m = _mm_load_sd(mid)
    h = _mm_load_sd(hi)

    t1 = _mm_min_pd(l, m)
    t2 = _mm_max_pd(l, m)
    t3 = _mm_min_pd(h, t2)

    t2 = _mm_max_pd(t1, t3)
    return _mm_cvtsd_f64(t2)

    # p = max(min(lo[0], mid[0]), min(max(lo[0], mid[0]), hi[0]))

    # a = min(lo[0], mid[0])
    #
    # b = max(lo[0], mid[0])
    # c = min(hi[0], b)
    #
    # p = max(a, c)
    # return p

cdef inline double med3(double* lo, double* hi):
    cdef double* mid = lo + ((hi - lo) >> 1) # n // 2

    if mid[0] < lo[0]:
        swap(lo, mid)
    if hi[0] < lo[0]:
        swap(lo, hi)
    if mid[0] < hi[0]:
        swap(mid, hi)
    return hi[0]

cdef inline double* _med3(double* lo, double* hi):
    """ 
    Choose pivot by median of 3 approach:
    median of first, middle and last elements
    :return: pivot pointer

    """
    cdef:
        size_t n = hi - lo
        double* mid = lo + (n >> 1) # n // 2

    if lo[0] < mid[0]:
        # a[0] < a[mid]
        if hi[0] > mid[0]:
            # median is a[mid]
            return mid
        elif hi[0] < lo[0]:
            # median is a[0]
            return lo
        else:
            # median is a[hi]
            return hi
    else:
        # a[mid] <= a[0]
        if hi[0] > lo[0]:
            # median is a[0]
            return lo
        elif hi[0] < mid[0]:
            # median is a[mid]
            return mid
        else:
            # median is a[hi]
            return hi


cdef inline size_t max_st(size_t a, size_t b):
    if a > b:
        return a
    else:
        return b


cdef:
    void err_exit(char* err_msg)
    void print_arr(double * arr, size_t n)