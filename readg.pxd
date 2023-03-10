from array_c cimport array_c
from graph cimport graph_c

cdef:
    (size_t, size_t) str2int(char* buf)
    (size_t, size_t, size_t) read_edge(char * buf)
    array_c * read_array(str filename)
    graph_c * read_graph(str filename)
    (graph_c *, graph_c *) read_graphs(str filename)
    graph_c* read_graph_l(str filename)
