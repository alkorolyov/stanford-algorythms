

from array_c cimport array_c
from graph cimport graph_c

cdef:
    array_c* topsort(graph_c * g)