from libc.stdio cimport printf
cimport numpy as np

cdef void err_exit(char* err_msg):
    printf(err_msg)
    exit(1)

cdef (double*, size_t) read_numpy(np.ndarray[double] arr):
    cdef:
        np.npy_intp *dims
        double *data
    if arr.flags['C_CONTIGUOUS']:
        dims = np.PyArray_DIMS(arr)
        data = <double*>np.PyArray_DATA(arr)
        return data, <size_t>dims[0]
    else:
        print('Array is non C-contiguous')
        exit(1)

