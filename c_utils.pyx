#cython: language_level=3

from libc.stdio cimport printf

cdef void err_exit(char* err_msg):
    printf(err_msg)
    exit(1)

