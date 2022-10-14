# cython: language_level=3
from array_c cimport array_c

ctypedef struct node_c:
    bint        explored
    size_t      leader
    size_t      fin_time
    array_c*    adj             # array of adjacent vertices

ctypedef struct graph_c:
    size_t      len
    node_c**    node

cdef:
    graph_c* create_graph_c(size_t n)
    void add_edge(graph_c* g, size_t v1, size_t v2)
    graph_c* dict2graph(dict graph)
    void free_graph(graph_c *g)
    void print_graph(graph_c *g, size_t length=*)
    void print_graph_ext(graph_c *g, size_t length=*)
    void mem_size(graph_c *g)
    dict rand_dict_graph(size_t n, size_t m, bint selfloops=*, bint directed=*)