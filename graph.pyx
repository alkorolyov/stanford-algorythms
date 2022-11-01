

from array_c cimport array_c, print_array, py2arr, create_arr, push_back_arr, resize_arr, free_arr
from libc.stdlib cimport rand, srand
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from utils import print_func_name


""" Graph structure in C """

cdef void print_mem(size_t * mem, size_t size):
    cdef size_t i
    for i in range(size):
        addr = hex(<size_t>(&mem[i]))
        val = hex(mem[i])
        print(f"{addr} : {val}")


cdef graph_c* create_graph_c(size_t n):
    """
    Create empty graph of size n
    :param n: size
    :return: pointer C graph
    """
    cdef graph_c* g
    cdef node_c* nd
    cdef size_t i

    g = <graph_c*> PyMem_Malloc(sizeof(graph_c))
    if g == NULL: exit(1)

    g.node = <node_c **> PyMem_Malloc(n * sizeof(node_c*))
    if g.node == NULL: exit(1)

    g.node[0] = <node_c *> PyMem_Malloc(n * sizeof(node_c))
    if g.node[0] == NULL: exit(1)

    g.len = n
    for i in range(n):
        nd = g.node[0] + i
        nd.explored = False
        nd.adj = NULL
        nd.len = NULL
        g.node[i] = nd
        # print(f"g.node[{i}]: ", hex(<size_t>g.node[i]))
        # print_mem(<size_t *> g.node[i], sizeof(node_c)/sizeof(size_t))

    # print_mem(<size_t*>g.node, g.len)
    return g

cdef void free_graph(graph_c *g):
    cdef size_t i
    cdef node_c* nd
    for i in range(g.len):
        nd = g.node[i]
        if nd.adj:
            free_arr(nd.adj)
        if nd.len:
            free_arr(nd.len)

    PyMem_Free(g.node[0])
    PyMem_Free(g.node)
    PyMem_Free(g)


cdef void add_edge(graph_c* g, size_t v1, size_t v2, size_t length=0):
    """
    Adds edge [v1, v2] to the directed graph G
    :param g: graph_c G
    :param v1: tail vertex
    :param v2: head vertex
    :param length: edge length
    :return: void
    """
    cdef node_c* nd = g.node[v1]

    if not nd.adj:
        nd.adj = create_arr(4)
    push_back_arr(nd.adj, v2)

    if length:
        if not nd.len:
            nd.len = create_arr(4)
        push_back_arr(nd.len, length)

cdef graph_c* dict2graph(dict graph):
    """
    Create C graph from python dict. Dict should have
    iterable adj list per each vertex key. With no missing
    vertices. 
    :param graph: python graph dict 
    :return: pointer to C graph
    """
    cdef:
        graph_c* g = create_graph_c(len(graph))
        node_c* nd
        size_t i = 0

    for i in range(g.len):
        nd = g.node[i]
        if graph[i]:
            nd.adj = py2arr(graph[i])
    # print_mem(<size_t*>g.node[0], g.len*5)
    return g

cdef graph_c* reverse_graph(graph_c* g):
    cdef:
        graph_c* g_rev
        size_t i, j, v1, v2
        node_c* nd
        array_c* orig
    g_rev = create_graph_c(g.len)
    for v1 in range(g.len):
        nd = g.node[v1]
        if nd.adj:
            orig = nd.adj
            for j in range(orig.size):
                v2 = orig.items[j]
                add_edge(g_rev, v2, v1)
    return g_rev

cdef void unexplore_graph(graph_c* g):
    cdef:
        size_t i
        node_c* nd

    for i in range(g.len):
        nd = g.node[i]
        nd.explored = False


cdef void print_graph(graph_c *g, size_t length=-1):
    cdef:
        size_t i
        node_c* nd

    for i in range(min(g.len, length)):
        nd = g.node[i]
        print(i, end=": ")
        if nd.adj:
            print_array(nd.adj)
        else:
            print("[]")


cdef void print_graph_ext(graph_c *g, size_t length=-1):
    cdef:
        size_t i, n
        node_c* nd

    for i in range(min(g.len, length)):
        nd = g.node[i]
        indent = len(str(i)) + 2

        print(i, end=": ")
        if nd.adj:
            print_array(nd.adj)
        else:
            print("[]")
        if nd.len:

            print(" " * indent, end="")
            print_array(nd.len)

        print(" " * (indent - 1), "exp:   ", nd.explored)
        # print("   leader:", nd.leader)


cdef void mem_size(graph_c *g):
    cdef:
        size_t i
        size_t mem_size = 0
    mem_size += g.len * (sizeof(node_c) + sizeof(node_c*))

    for i in range(g.len):
        mem_size += g.node[i].adj.size * sizeof(array_c)
    print("c size: ", mem_size)


cdef dict rand_dict_graph(size_t n, size_t m,
                          bint selfloops=False,
                          bint directed=True):
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

cdef graph_c* rand_graph_l(size_t n, size_t m, size_t seed=0):
    """
    Random directed graph with distances
    :param n: nodes
    :param m: edges
    :param seed: (optional) random seed
    
    :return: C graph
    """
    cdef:
        size_t i, j, v1, v2, l
        graph_c* g = create_graph_c(n)

    if seed:
        srand(seed)

    for i in range(m):
        v1 = rand() % n
        v2 = rand() % n
        l = rand() % n
        add_edge(g, v1, v2, l)
    return g




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
    assert g.node[0].adj.size == 2
    assert g.node[1].adj.size == 1
    assert g.node[2].adj == NULL
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
    for i in range(arr.size):
        assert arr.items[i] == graph[0][i]

    free_graph(g)


def test_dict2graph_2():
    print_func_name()
    graph = {0: [1, 2],
             1: [],
             2: [0]}
    cdef graph_c* g = dict2graph(graph)
    cdef array_c * arr = g.node[0].adj
    for i in range(arr.size):
        assert arr.items[i] == graph[0][i]

    assert g.node[1].adj == NULL

    arr = g.node[2].adj
    for i in range(arr.size):
        assert arr.items[i] == graph[2][i]


    free_graph(g)


def test_dict2graph_random():
    print_func_name()
    cdef:
        graph_c* g
        array_c * arr
        size_t i, j
    for i in range(100):
        graph = rand_dict_graph(50, 250)
        g = dict2graph(graph)
        for key in graph:
            arr = g.node[key].adj
            for j, val in enumerate(graph[key]):
                assert arr.items[j] == val
        free_graph(g)


def test_reverse_graph():
    print_func_name()
    cdef:
        graph_c* g
        graph_c* r
    graph = {0: [1, 2],
             1: [],
             2: []}
    g = dict2graph(graph)
    r = reverse_graph(g)
    assert r.node[0].adj == NULL
    assert r.node[1].adj.items[0] == 0
    assert r.node[2].adj.items[0] == 0
    free_graph(g)
    free_graph(r)
