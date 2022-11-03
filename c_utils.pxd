from libc.stdlib cimport rand, srand
cimport numpy as np
from time import time
# from libc.time cimport time

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
        Compute a pseudorandom integer.
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




cdef:
    void err_exit(char* err_msg)
    (double *, size_t) read_numpy(np.ndarray[double] arr)