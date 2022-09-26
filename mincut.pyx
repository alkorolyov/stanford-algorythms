# cython: language_level=3

# cython: profile=True
# cython: linetrace=True
# cython: binding=True

# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# distutils: extra_compile_args = /O2 /Ob3 /arch:AVX2 /openmp
import random

import cython
import numpy as np
cimport numpy as cnp

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc, free, rand
from libc.math cimport sqrt
from pprint import pprint
from time import time

cdef size_t C_MALLOC = 0
cdef size_t PY_MALLOC = 1

@cython.wraparound
def read_file() -> dict:
    with open("kargerMinCut.txt", "r") as f:
        # print(f.read().split("\n")[:-1])
        lines = [s for s in f.read().split("\n")[:-1]]
    graph = {}
    for l in lines:
        # vertices = np.array([int(s) for s in l.split("\t")[:-1]], dtype=np.uint16)
        vertices = [int(s) for s in l.split("\t")[:-1]]
        graph[vertices[0]] = vertices[1:]
    return graph

def max_degree(graph: dict) -> int:
    max_d = 0
    for v in graph.values():
        max_d = max(len(v), max_d)
    return max_d

def get_size(graph: dict) -> int:
    size = 0
    for v in graph.values():
        size += len(v)
    return size


ctypedef struct node_c:
    size_t  vertex
    size_t  len
    size_t* next


ctypedef struct graph_c:
    size_t   mem_mode
    size_t   len
    node_c*  node
    size_t   buff_len
    size_t*  buff


cdef void print_debug(graph_c *g):
    print("size_t:  ", sizeof(size_t), "bytes")
    print("buff:    ", hex(<size_t>g.buff))
    print("g.node:  ", hex(<size_t>g.node))

    cdef size_t* nd
    nd = <size_t*>g.node
    print("g.node:")
    for i in range(g.len):
        for j in range(3):
            if j < 2:
                print(nd[i*3 + j], end=", ")
            else:
                print(hex(nd[i*3 + j]))

    print("g.node[i].next[j]:")
    for i in range(g.len):
        print(i, ":", end="[")
        for j in range(g.node[i].len):
            print(g.node[i].next[j], end="")
            if j < g.node[i].len - 1:
                print(end=", ")
            else:
                print("]")

    print("Buffer:")
    for i in range(g.len):
        print(i, ":", end="[")
        for j in range(g.buff_len//g.len):
            print(g.buff[i*(g.buff_len//g.len) + j], end="")
            if j < g.buff_len//g.len - 1:
                print(end=", ")
            else:
                print("]")

    print("==============")


cdef graph_c* create_graph_c():
    cdef graph_c* g
    cdef size_t i, j, n

    g = <graph_c*> malloc(sizeof(graph_c))
    g.len = 3
    g.node = <node_c *> malloc(g.len * sizeof(node_c))
    g.buff_len = g.len * 4
    g.buff = <size_t *> malloc(g.buff_len * sizeof(size_t))

    for j in range(g.buff_len):
        g.buff[j] = 0

    for i in range(g.len):
        g.node[i].vertex = i + 1
        g.node[i].len = 3
        g.node[i].next = g.buff + i * 4
        g.node[i].next[i] = 1

    # print_debug(g)

    return g


cdef void free_graph(graph_c *g):
    if g.mem_mode == C_MALLOC:
        free(g.buff)
        free(g.node)
        free(g)
    elif g.mem_mode == PY_MALLOC:
        PyMem_Free(g.buff)
        PyMem_Free(g.node)
        PyMem_Free(g)

cdef graph_c* alloc_graph(graph_c *g_old):
    cdef graph_c * g_new

    if g_old.mem_mode == C_MALLOC:
        g_new = <graph_c *> malloc(sizeof(graph_c))
        g_new.node = <node_c *> malloc(g_old.len * sizeof(node_c))
        g_new.buff = <size_t *> malloc(g_old.buff_len * sizeof(size_t))
    elif g_old.mem_mode == PY_MALLOC:
        g_new = <graph_c *> PyMem_Malloc(sizeof(graph_c))
        g_new.node = <node_c *> PyMem_Malloc(g_old.len * sizeof(node_c))
        g_new.buff = <size_t *> PyMem_Malloc(g_old.buff_len * sizeof(size_t))

    g_new.len = g_old.len
    g_new.mem_mode = g_old.mem_mode
    g_new.buff_len = g_old.buff_len
    return g_new

cdef void copy_graph(graph_c* g_new, graph_c *g_old):
    cdef node_c* node
    cdef size_t i = 0

    g_new.len = g_old.len
    g_new.mem_mode = g_old.mem_mode
    g_new.buff_len = g_old.buff_len

    for i in range(g_new.len):
        node = &g_new.node[i]
        node.vertex = g_old.node[i].vertex
        node.len = g_old.node[i].len
        # base addr
        node.next = g_new.buff + i * (g_new.buff_len // g_new.len)
        # copy adj list
        if node.len == 0:
            continue
        for j in range(node.len):
            node.next[j] = g_old.node[i].next[j]

cdef void print_graph(graph_c *g):
    cdef size_t n = g.len
    cdef node_c* nd
    if g.len == 0:
        print("[]")
        return

    for i in range(n):
        nd = &g.node[i]
        print(f"{nd.vertex}: [", end="")

        if nd.len == 0:
            print("]")
            continue

        for j in range(nd.len):
            print(nd.next[j], end="")
            if  j == nd.len - 1:
                print("]")
            else:
                print(", ", end="")



cdef graph_c* read_graph_c(dict graph, size_t mem_mode=C_MALLOC):
    cdef graph_c* g
    cdef size_t j, buff_len, g_len

    g_len = len(graph)
    buff_len = max_degree(graph) * g_len * g_len

    if mem_mode == C_MALLOC:
        g = <graph_c*> malloc(sizeof(graph_c))
        g.node = <node_c *> malloc(g_len * sizeof(node_c))
        g.buff = <size_t *> malloc(buff_len * sizeof(size_t))

    elif mem_mode == PY_MALLOC:
        g = <graph_c*> PyMem_Malloc(sizeof(graph_c))
        g.node = <node_c *> PyMem_Malloc(g_len * sizeof(node_c))
        g.buff = <size_t *> PyMem_Malloc(buff_len * sizeof(size_t))


    g.len = g_len
    g.mem_mode = mem_mode
    g.buff_len = buff_len

    cdef node_c* node
    cdef size_t i = 0

    for key in graph:
        node = &g.node[i]
        node.vertex = key
        node.len = len(graph[key])
        # base addr
        node.next = g.buff + i * (g.buff_len // g.len)

        # print("buff addr:", hex(<size_t> g.buff))
        # print("g.node[i] offset:", hex(<size_t>(g.node[i].next - g.buff)))

        # print("node.vertex", node.vertex)
        # print(f"g.node[{i}].vertex", g.node[i].vertex)
        # print("node.len", node.len)
        # print(f"g.node[{i}].len", g.node[i].len)


        # print("key", key)
        # copy adj list
        if node.len == 0:
            i += 1
            continue
        for j in range(node.len):
            # print("current size:", (i * (g.buff_len // g.len) + j) * sizeof(size_t))
            # print("idx:", (i * (g.buff_len // g.len) + j))
            # print(f"py graph:[{key}][{j}] ", graph[key][j])
            node.next[j] = graph[key][j]
            # g.node[i].next[j] = graph[key][j]
            # print(f"g.node[{i}].next[{j}] offset:", <size_t>(&g.node[i].next[j] - g.buff)//sizeof(size_t))
            # print(f"buff[{node.vertex}][{j}]: ", g.buff[i * (g.buff_len // g.len) + j])

        # print("===================")
            # g.node[i].next[j] = 0

        # for j in range(node.len):
            # print(f"c graph[{node.vertex}][{j}]: ", node.next[j])
            # print(f"buff[{node.vertex}][{j}]: ", g.buff[i * (g.buff_len // g.len) + j])

        i += 1

    # print_debug(g)

    return g

cdef size_t idx_from_value(graph_c *g, size_t val):
    cdef size_t i
    for i in range(g.len):
        if g.node[i].vertex == val:
            return i

cdef (size_t, size_t) random_pair(graph_c *g):
    """    
    :param g: input graph 
    :return: random vertex pairs, values
    """
    cdef:
        size_t num_pairs = 0
        size_t i, n, p, v_idx
        size_t j = 0

    n = g.len
    for i in range(n):
        num_pairs += g.node[i].len
    p = rand() % num_pairs

    # debug
    # print_graph(g)
    # print("p", p)

    cdef node_c* node
    for i in range(n):
        node = &g.node[i]
        j += node.len
        # print("i", i, "j", j)
        if j > p:
            v_idx = p - j + node.len
            # print("i", i)
            return node.vertex, node.next[v_idx]


cdef void _pop_from_arr(size_t idx, size_t *a, size_t n):
    """
    Pops element from array with left shift.
    :param idx: index to remove 
    :param a: array
    :param n: length
    """
    for i in range(idx, n - 1):
        a[i] = a[i + 1]


cdef void _pop_from_node(node_c *nd, size_t val):
    """
    Pops multiple elements by value from single node in place.
    :param val: value to remove
    :param nd: node with adjacency list for single vertex 
    """
    # cdef size_t i = 0
    # cdef size_t n = nd.len
    # while i < n:
    #     if nd.next[i] == val:
    #         _pop_from_arr(i, nd.next, n) # a[i] = a[i + 1] so no need to increase counter
    #         n -= 1
    #     else:
    #         i += 1
    # nd.len = n


    cdef size_t i
    cdef size_t j = 0
    for i in range(nd.len):
        if nd.next[i] != val:
            if i != j:
                nd.next[j] = nd.next[i]
            j += 1
    nd.len = j


cdef void delete_self_loops(graph_c *g, size_t idx1, size_t idx2):
    """
    Deletes cross references of idx1 and idx2 vertices, which forms self loops.
    :param g: graph
    :param idx1: index
    :param idx2: index
    """
    _pop_from_node(&g.node[idx2], g.node[idx1].vertex)
    _pop_from_node(&g.node[idx1], g.node[idx2].vertex)

cdef void transfer_vertices(graph_c *g, size_t dest, size_t source):
    """    
    :param g: input graph
    :param dest: index 
    :param source: index
    :return: 
    """
    cdef size_t base, i
    for i in range(g.node[source].len):
        base = g.node[dest].len
        g.node[dest].next[base + i] = g.node[source].next[i]
    g.node[dest].len += g.node[source].len


cdef void delete_vertex(size_t idx, graph_c *g):
    for i in range(idx, g.len - 1):
        # print("i", i)
        g.node[i].vertex = g.node[i + 1].vertex
        g.node[i].next = g.node[i + 1].next
        g.node[i].len = g.node[i + 1].len
    g.len -= 1


cdef void replace_references(size_t new_idx, size_t old_idx, graph_c *g):
    cdef size_t i, j
    cdef node_c* nd
    for i in range(g.len):
        nd = &g.node[i]
        for j in range(nd.len):
            if nd.next[j] == g.node[old_idx].vertex:
                nd.next[j] = g.node[new_idx].vertex



cdef void contract(graph_c *g):
    cdef size_t i, j
    
    i, j = random_pair(g)
    # print("(i, j)", i, j)

    i = idx_from_value(g, i)
    j = idx_from_value(g, j)

    delete_self_loops(g, i, j)
    transfer_vertices(g, i, j)
    replace_references(i, j, g)
    delete_vertex(j, g)


cdef size_t _mincut(graph_c *g):
    while g.len > 2:
        contract(g)
    return g.node[0].len

cpdef size_t mincut_n(dict graph, size_t N, mem_mode=C_MALLOC):
    cdef graph_c *g
    cdef graph_c *g_orig
    cdef size_t i
    cdef size_t minc, cut

    g_orig = read_graph_c(graph, mem_mode)
    g = alloc_graph(g_orig)
    minc = g_orig.len * (g_orig.len - 1) // 2
    start_time = time()
    for i in range(N):
        copy_graph(g, g_orig)
        # print("before:")
        # print_graph(g)
        # if i % 1000 == 0:
        #     print(f"{i} / {N}: {time() - start_time:.1f}s")

        cut = _mincut(g)
        # print("after:")
        # print_graph(g)
        # print("================")
        if cut < minc:
            minc = cut
            # print("mincut:", minc)

    free_graph(g)
    free_graph(g_orig)

""" #############################################################
    ###################### UNIT TESTS ###########################
    ############################################################# 
"""
from utils import print_func_name

cdef void assert_buff(graph_c *g):
    cdef size_t i, j
    for i in range(g.len):
        if g.node[i].len == 0:
            continue
        for j in range(g.node[i].len):
            assert g.buff[i * (g.buff_len // g.len) + j] == g.node[i].next[j]

def gen_random_graph(n, m, selfloops: bool = False):
    graph = {}
    for i in range(1, n + 1):
        graph[i] = []

    for j in range(m):
        v1 = random.randrange(1, n + 1)
        v2 = random.randrange(1, n + 1)
        if not selfloops and v1 == v2:
            continue
        graph[v1].append(v2)
        graph[v2].append(v1)

    # for i in range(n):
    #     graph[i] = [random.randrange(n)]
    #     for j in range(random.randrange(m //2, m)):
    #         graph[i].append(random.randrange(n))
    return graph

def test_create_graph():
    print_func_name()
    cdef graph_c* g
    g = create_graph_c()
    # print_graph(g)
    free_graph(g)

def test_replace_references():
    print_func_name()
    cdef graph_c *g
    cdef size_t n

    graph = {1: [2, 2],
             2: []}

    g = read_graph_c(graph)
    replace_references(0, 1, g)
    assert g.node[0].next[0] == 1
    assert g.node[0].next[1] == 1
    # print_graph(g)
    free_graph(g)


def test_delete_vertex():
    print_func_name()
    cdef graph_c *g

    graph = {1: [2, 1],
             2: [1, 3],
             3: [3, 4]}

    g = read_graph_c(graph)
    # print_graph(g)
    delete_vertex(0, g)
    assert g.node[0].next[0] == 1
    assert g.node[1].next[0] == 3
    # print_graph(g)
    free_graph(g)

    g = read_graph_c(graph)
    delete_vertex(1, g)
    assert g.node[0].next[0] == 2
    assert g.node[1].next[0] == 3
    # print_graph(g)
    free_graph(g)

    g = read_graph_c(graph)
    delete_vertex(2, g)
    assert g.node[0].next[0] == 2
    assert g.node[1].next[0] == 1
    # print_graph(g)
    free_graph(g)


def test_delete_vertex_1():
    print_func_name()
    cdef graph_c *g

    graph = {1: [],
             2: []}

    g = read_graph_c(graph)
    delete_vertex(0, g)
    assert g.len == 1
    assert g.node[0].vertex == 2
    # print_graph(g)
    free_graph(g)


def test_transfer_vertices():
    print_func_name()
    cdef graph_c *g

    g = read_graph_c({1: [1, 2, 3],
                       2: [4, 5, 6]})
    # print_graph(g, n)
    transfer_vertices(g, 0, 1)
    assert g.node[0].next[0] == 1
    assert g.node[0].next[1] == 2
    assert g.node[0].next[2] == 3
    assert g.node[0].next[3] == 4
    assert g.node[0].next[4] == 5
    assert g.node[0].next[5] == 6
    assert g.node[0].len == 6

    # print_graph(g, n)

def test_pop_from_graph():
    print_func_name()
    cdef graph_c* g

    g = read_graph_c({1: [2, 2, 2, 3],
                       2: [3, 1, 1, 1]})

    _pop_from_node(&g.node[0], 2)
    _pop_from_node(&g.node[1], 1)

    assert g.node[0].len == 1
    assert g.node[1].len == 1
    assert g.node[0].next[0] == 3
    assert g.node[1].next[0] == 3

    free(g.node)
    free(g.buff)

def test_pop_from_graph_1():
    print_func_name()
    cdef graph_c* g

    g = read_graph_c({1: [2, 2]})

    _pop_from_node(&g.node[0], 2)

    assert g.node[0].len == 0

    free_graph(g)

def test_delete_self_loops():
    print_func_name()

    cdef graph_c * g
    graph = {1: [2, 2, 2, 3],
             2: [3, 1, 1, 1]}

    g = read_graph_c(graph)
    delete_self_loops(g, 0, 1)

    assert g.node[0].len == 1
    assert g.node[1].len == 1
    assert g.node[0].next[0] == 3
    assert g.node[1].next[0] == 3

    free_graph(g)

def test_random_pair():
    print_func_name()
    cdef graph_c* g
    cdef size_t j, k
    graph = {1: [2, 3, 4],
             2: [1, 3],
             3: [2, 1, 4],
             4: [1, 3]}
    g = read_graph_c(graph)
    for i in range(1000):
        j, k = random_pair(g)
        assert j != k
        assert j in graph.keys()
        assert k in graph.keys()
    free_graph(g)

def test_read_graph_c_1():
    print_func_name()
    cdef graph_c* g
    graph = {1: [],
             2: [3, 4, 5]}
    g = read_graph_c(graph)
    assert_buff(g)
    assert g.len == 2
    assert g.node[1].next[0] == 3
    assert g.node[1].next[1] == 4
    assert g.node[1].next[2] == 5
    free_graph(g)


def test_read_graph_c_2():
    print_func_name()
    cdef graph_c* g
    graph = {1: [1, 2],
             2: []}
    g = read_graph_c(graph)
    assert_buff(g)
    assert g.len == 2
    assert g.node[0].next[0] == 1
    assert g.node[0].next[1] == 2
    free_graph(g)


def test_read_graph_c_3():
    print_func_name()
    cdef graph_c* g
    graph = {1: [],
             2: []}
    g = read_graph_c(graph)
    assert_buff(g)
    free_graph(g)


def test_read_graph_c_4():
    print_func_name()
    cdef graph_c* g
    graph = {1: [1, 2, 3],
             2: [3, 4, 5]}
    g = read_graph_c(graph)
    assert_buff(g)
    assert g.len == 2
    assert g.node[0].next[0] == 1
    assert g.node[0].next[1] == 2
    assert g.node[0].next[2] == 3
    assert g.node[1].next[0] == 3
    assert g.node[1].next[1] == 4
    assert g.node[1].next[2] == 5
    free_graph(g)


def test_read_graph_c_random():
    print_func_name()
    cdef graph_c* g

    for i in range(100):
        graph = gen_random_graph(10, 200)
        g = read_graph_c(graph)
        for key in graph:
            for idx, val in enumerate(graph[key]):
                # find key in C graph
                for i in range(g.len):
                    if key == g.node[i].vertex:
                        break

                # if val != g.node[i].next[idx]:
                #     for key in graph:
                #         print(f"{key}: {graph[key]}")
                #     print("=============")
                #
                #     print_graph(g)
                #     print("=============")
                #
                #     print("v, idx", g.node[i].vertex, idx)

                assert val == g.node[i].next[idx]

        assert_buff(g)
        free_graph(g)

def test_copy_graph():
    print_func_name()
    cdef graph_c* g
    cdef graph_c* g_copy

    # graph = {1: [1],
    #          2: [2]}
    graph = gen_random_graph(5, 5)
    g = read_graph_c(graph)
    g_copy = alloc_graph(g)
    copy_graph(g_copy, g)

    cdef size_t i, j
    for i in range(g.len):
        if g.node[i].len == 0:
            continue
        for j in range(g.node[i].len):
            assert g.node[i].next[j] == g_copy.node[i].next[j]

    free_graph(g)
    free_graph(g_copy)


def test_contract():
    print_func_name()
    cdef graph_c* g

    graph = {1: [2, 3, 4],
             2: [1, 3],
             3: [2, 1, 4],
             4: [1, 3]}

    g = read_graph_c(graph)
    contract(g)
    print_graph(g)
    free_graph(g)

def test_mincut():
    print_func_name()
    graph = {1: [2, 3, 4],
             2: [1, 3],
             3: [2, 1, 4],
             4: [1, 3]}
    g = read_graph_c(graph)
    _mincut(g)
    print_graph(g)
    free_graph(g)

def test_mincut_1():
    print_func_name()
    graph = gen_random_graph(200, 3000)
    # print(graph)
    g = read_graph_c(graph)
    # print_graph(g)
    # print("===================================")
    _mincut(g)
    # print_graph(g)
    assert g.len == 2
    cdef size_t i
    for i in range(g.node[0].len):
        assert g.node[0].next[i] == g.node[1].vertex
        assert g.node[1].next[i] == g.node[0].vertex
    free_graph(g)

def test_mincut_N():
    print_func_name()
    graph = gen_random_graph(10, 20)
    # graph = {1: [2, 3, 4],
    #          2: [1, 3],
    #          3: [2, 1, 4],
    #          4: [1, 3]}
    mincut_n(graph, 5)