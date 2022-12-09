

cimport numpy as cnp
import cython

cnp.import_array()

from time import time

import numpy as np
import os

from readg cimport read_graphs, read_graph
from array_c cimport array_c, free_arr, reverse_arr
from topsort cimport topsort
from graph cimport graph_c, node_c, dict2graph, reverse_graph, free_graph, print_graph, print_graph_ext
from dfs cimport dfs_ordered_loop

from utils import print_func_name



cdef void print_mem(size_t * mem, size_t size):
    cdef size_t i
    for i in range(size):
        addr = hex(<size_t>(&mem[i]))
        val = hex(mem[i])
        print(f"{addr} : {val}")


cdef void scc(graph_c* g, graph_c* g_rev, bint debug=False, bint timeit=False):
    cdef:
        size_t i
        node_c* nd
        array_c* order

    if debug:
        print_graph(g)
        print("=======")
        print_graph(g_rev)
        print("=== DFS g_rev ====")

    if timeit:
        print(f"{'Running topsort(g_rev)':58s}", end="")
        start_time = time()

    order = topsort(g_rev)
    reverse_arr(order)

    if timeit:
        print(f"{time() - start_time:.2f}s ")

    if debug:
        print("g_rev.len:", g_rev.len)
        print_graph_ext(g_rev)


    if debug:
        print("===== DFS ordered loop =====")

    if timeit:
        print(f"{'Running dfs_ordered_loop(g)':58s}", end="")
        start_time = time()

    dfs_ordered_loop(g, order)
    free_arr(order)

    if timeit:
        print(f"{time() - start_time:.2f}s ")

    if debug:
        print_graph_ext(g)

@cython.wraparound
def scc_py(str filename):
    cdef:
        size_t i
        graph_c * g
        graph_c * g_rev


    g = read_graph(filename)
    # g, g_rev = read_graphs(filename)
    g_rev = reverse_graph(g)

    scc(g, g_rev, debug=False)

    l = np.empty(g.len, dtype=np.uint64)
    cdef size_t [:] l_view = l
    for i in range(g.len):
        l_view[i] = g.node[i].leader

    free_graph(g)
    free_graph(g_rev)

    val, cnt = np.unique(l, return_counts=True)
    res = np.flip(np.sort(cnt)[-5:])
    return res


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """

TEST_PATH = "tests//course2_assignment1SCC//"

def test_scc_1():
    
    graph = {0: [],
             1: [],
             2: [0, 1]}
    cdef:
        graph_c* g = dict2graph(graph)
        graph_c* g_rev = reverse_graph(g)

        size_t i

    scc(g, g_rev, debug=False)
    assert g.node[0].leader == 0
    assert g.node[1].leader == 1
    assert g.node[2].leader == 2


def test_scc_2():
    
    graph = {0: [1, 3],
             1: [0],
             2: [3],
             3: [2]}
    cdef:
        graph_c* g = dict2graph(graph)
        graph_c * g_rev = reverse_graph(g)
        size_t i

    scc(g, g_rev)
    assert g.node[0].leader == 0
    assert g.node[1].leader == 0
    assert g.node[2].leader == 2
    assert g.node[3].leader == 2

def test_scc_3():
    
    graph = {0: [1, 3],
             1: [0],
             2: [3],
             3: [2, 0]}
    cdef:
        graph_c* g = dict2graph(graph)
        graph_c * g_rev = reverse_graph(g)
        size_t i

    scc(g, g_rev)
    assert g.node[0].leader == 0
    assert g.node[1].leader == 0
    assert g.node[2].leader == 0
    assert g.node[3].leader == 0


def test_scc_4():
    
    graph = {0: [1],
             1: [0, 2],
             2: [3],
             3: [4],
             4: [2]}
    cdef:
        graph_c* g = dict2graph(graph)
        graph_c * g_rev = reverse_graph(g)
        size_t i
        node_c* nd

    scc(g, g_rev)

    l = np.empty(g.len, dtype=np.uint64)
    cdef size_t [:] l_view = l
    for i in range(g.len):
        nd = g.node[i]
        l_view[i] = nd.leader

    # print(l)
    val, cnt = np.unique(l, return_counts=True)

    assert np.sort(cnt)[0] == 2
    assert np.sort(cnt)[1] == 3
    # print(np.qsort(cnt))

# @cython.wraparound
# def test_scc_big():
#     cdef:
#         size_t i
#         graph_c * g
#         graph_c * g_rev
#
#     print(f"\n{'Running read_graphs()':58s}", end="")
#     start = time()
#
#     g, g_rev = read_graphs("scc.txt")
#
#     print(f"{time() - start:.2f}s ")
#
#     scc(g, g_rev, debug=False, timeit=True)
#
#     # print_g_ext(g, 100)
#
#     print(f"{'Running np.unique(leaders)':58s}", end="")
#     start = time()
#
#     l = np.empty(g.len, dtype=np.uint64)
#     cdef size_t [:] l_view = l
#     for i in range(g.len):
#         l_view[i] = g.node[i].leader
#
#     val, cnt = np.unique(l, return_counts=True)
#
#     print(f"{time() - start:.2f}s ")
#
#     # print(np.flip(np.sort(cnt)[-5:]))
#
#     free_graph(g)
#     free_graph(g_rev)

def _test_single_case(fname="input_mostlyCycles_1_8.txt"):
    cdef size_t i    
    input_fname = TEST_PATH + fname
    output_fname = TEST_PATH + "output" + fname.split("input")[1]
    res = scc_py(input_fname)
    with open(output_fname, "r") as f:
        correct = [int(s) for s in f.read().replace("\n", "").split(",")]
        for i in range(len(res)):
            assert correct[i] == res[i]

def test_single_case():
    _test_single_case()

# def test_all_casses():
#     for f in os.listdir(TEST_PATH):
#         if "input" in f:
#             _test_single_case(f)


