# cython: language_level=3

from graph cimport graph_c
from stack cimport stack_c
from array_c cimport array_c


cdef:
    void dfs_rec(graph_c* g, size_t s, array_c* out=*)
    void dfs_stack(graph_c * g, size_t s, array_c * out=*)
    void dfs_ordered_loop(graph_c * g, array_c * order)