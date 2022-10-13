# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

from array_c cimport array_c, print_array, list2arr, create_arr, resize_arr, free_arr
from libc.stdlib cimport malloc, free, rand
from utils import print_func_name


""" Graph structure in C """

ctypedef struct node_c:
    bint        explored
    size_t      leader
    size_t      fin_time
    size_t      degree          # total number of adjacent vertices
    array_c*    adj             # array of adjacent vertices

ctypedef struct graph_c:
    size_t      len
    node_c**    node


cdef graph_c* create_graph_c(size_t n):
    """
    Create empty graph of size n
    :param n: size
    :return: pointer C graph
    """
    cdef graph_c* g
    cdef node_c* nd
    cdef size_t i

    g = <graph_c*> malloc(sizeof(graph_c))
    g.len = n
    g.node = <node_c **> malloc(g.len * sizeof(node_c*))
    g.node[0] = <node_c *> malloc(g.len * sizeof(node_c))

    for i in range(n):
        nd = g.node[0] + i
        nd.degree = 0
        nd.explored = False
        nd.fin_time = -1
        nd.adj = NULL
        g.node[i] = nd
        # print(f"g.node[{i}]: ", hex(<size_t>g.node[i]))

    # print_mem(<size_t*>g.node, g.len)

    return g

cdef inline void _add_edge(node_c* nd, size_t v):
    cdef size_t i
    i = nd.degree
    nd.adj.items[i] = v
    nd.degree += 1

cdef void add_edge(graph_c* g, size_t v1, size_t v2):
    cdef node_c* nd = g.node[v1]
    if nd.adj == NULL:
        nd.adj = create_arr(4)
    elif nd.degree == nd.adj.maxsize:
        resize_arr(nd.adj)
    _add_edge(nd, v2)

cdef graph_c* dict2graph(dict graph):
    """
    Create C graph from python dict.
    :param graph: graph in standardized form (sorted vertices 0 .. n - 1)
    :return: pointer to C graph
    """
    cdef:
        graph_c* g = create_graph_c(len(graph))
        node_c* nd
        size_t i = 0

    for i in range(g.len):
        nd = g.node[i]
        nd.degree = len(graph[i])
        if graph[i]:
            nd.adj = list2arr(graph[i])
    # print_mem(<size_t*>g.node[0], g.len*5)
    return g

cdef void free_graph(graph_c *g):
    cdef size_t i
    cdef node_c* nd
    for i in range(g.len):
        nd = g.node[i]
        if nd.adj:
            free_arr(nd.adj)
    free(g.node[0])
    free(g.node)
    free(g)

cdef void print_graph(graph_c *g, size_t length=-1):
    cdef:
        size_t i
        node_c* nd

    for i in range(min(g.len, length)):
        nd = g.node[i]
        print(i, end=": ")
        print_array(nd.adj, nd.degree)


cdef void print_graph_ext(graph_c *g, size_t length=-1):
    cdef:
        size_t i, n
        node_c* nd

    for i in range(max(g.len, length)):
        nd = g.node[i]
        print(i, end=": ")
        print_array(nd.adj, nd.degree)
        print("   exp:   ", nd.explored)
        print("   ft:    ", hex(nd.fin_time))
        print("   leader:", nd.leader)


cdef void mem_size(graph_c *g):
    cdef:
        size_t i
        size_t mem_size = 0
    mem_size += g.len * (sizeof(node_c) + sizeof(node_c*))

    for i in range(g.len):
        mem_size += g.node[i].degree * sizeof(array_c)
    print("c size: ", mem_size)


cdef dict rand_dict_graph(size_t n, size_t m, bint selfloops=False, bint directed=True):
    cdef:
        size_t i, j, v1, v2

    graph = {}
    for i in range(n):
        graph[i] = []

    for j in range(m):
        v1 = rand() % n
        v2 = rand() % n
        if not selfloops and v1 == v2:
            continue
        graph[v1].append(v2)
        if not directed:
            graph[v2].append(v1)
    return graph


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_create_graph():
    print_func_name()
    cdef graph_c * g = create_graph_c(3)
    add_edge(g, 0, 1)
    add_edge(g, 0, 2)
    add_edge(g, 1, 2)
    # print_graph(g)
    assert g.node[0].adj.items[0] == 1
    assert g.node[0].adj.items[1] == 2
    assert g.node[1].adj.items[0] == 2
    assert g.node[2].adj == NULL
    assert g.node[0].degree == 2
    assert g.node[1].degree == 1
    assert g.node[2].degree == 0


    free_graph(g)

def test_add_edge():
    print_func_name()
    cdef graph_c * g = create_graph_c(2)

    cdef size_t i
    for i in range(1024):
        add_edge(g, 0, 1)

    free_graph(g)



def test_dict2graph():
    print_func_name()
    graph = {0: [1, 2],
             1: [],
             2: [1]}
    cdef graph_c* g = dict2graph(graph)

    cdef array_c* arr = g.node[0].adj
    assert arr.items[0] == 1
    assert arr.items[1] == 2

    assert g.node[1].adj == NULL

    arr = g.node[2].adj
    assert arr.items[0] == 1

    # print(graph)
    # print_graph(g)
    free_graph(g)


def test_dict2graph_1():
    print_func_name()
    graph = {0: [1, 2, 3]}
    cdef graph_c* g = dict2graph(graph)
    cdef node_c* nd = g.node[0]
    cdef array_c* arr = g.node[0].adj
    for i in range(nd.degree):
        assert arr.items[i] == graph[0][i]

    free_graph(g)


def test_dict2graph_2():
    print_func_name()
    graph = {0: [1, 2],
             1: [],
             2: [0]}
    cdef graph_c* g = dict2graph(graph)

    cdef node_c* nd = g.node[0]
    cdef array_c * arr = nd.adj
    for i in range(nd.degree):
        assert arr.items[i] == graph[0][i]

    free_graph(g)


def test_dict2graph_random():
    print_func_name()
    cdef graph_c* g
    cdef node_c * nd
    cdef array_c * arr
    cdef size_t i, j
    for i in range(100):
        graph = rand_dict_graph(50, 250)
        g = dict2graph(graph)
        for key in graph:
            nd = g.node[key]
            arr = nd.adj
            for j, val in enumerate(graph[key]):
                assert arr.items[j] == val
        free_graph(g)


