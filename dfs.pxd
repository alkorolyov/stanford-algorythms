# cython: language_level=3

from graph cimport graph_c
from stack cimport stack_c


cdef:
    void dfs_rec(graph_c* g, size_t s, stack_c* output=*, size_t* ft=*)
    void dfs_stack(graph_c * g, size_t s, stack_c * output=*, size_t * ft=*)