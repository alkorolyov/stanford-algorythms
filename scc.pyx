# cython: language_level=3

# cython: profile=False
# cython: linetrace=False
# cython: binding=False

# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

cimport numpy as cnp
cnp.import_array()

import numpy as np
import pickle
import sys
import random

from stack cimport stack_c, create_stack, push, pop, peek, \
    free_stack, size_s, print_stack, is_empty_s


from readg cimport read_graphs
from array_c cimport array_c, reverse_arr
from topsort cimport topsort
from graph cimport graph_c, node_c, dict2graph, reverse_graph, free_graph, print_graph, print_graph_ext
from dfs cimport dfs_stack, dfs_ordered_loop

from libc.stdlib cimport malloc, realloc, free, EXIT_FAILURE, rand, qsort
from utils import print_func_name
from tqdm import tqdm, trange



cdef void print_mem(size_t * mem, size_t size):
    cdef size_t i
    for i in range(size):
        addr = hex(<size_t>(&mem[i]))
        val = hex(mem[i])
        print(f"{addr} : {val}")



cdef void dfs_loop_1(graph_c* g_rev):
    cdef size_t i
    cdef size_t ft = 0

    for i in range(g_rev.len):
        if not g_rev.node[i].explored:
            dfs_stack(g_rev, i, NULL, &ft)

cdef void dfs_loop_2(graph_c* g, table* ft_table):
    cdef size_t i, j

    for i in range(g.len):
        j = ft_table[i].idx
        if not g.node[j].explored:
            dfs_stack(g, j)


cdef int cmp_table(const void *a, const void *b) nogil:
    cdef:
        table* nd1 = <table*>a
        table* nd2 = <table*>b
    if nd1.val > nd2.val:
        return -1
    elif nd1.val < nd2.val:
        return 1
    else:
        return 0

ctypedef struct table:
    size_t idx
    size_t val

cdef void scc(graph_c* g, graph_c* g_rev, bint debug=False):
    cdef:
        size_t i
        node_c* nd
        array_c* ft_order

    if debug:
        print_graph(g)
        print("=======")
        print_graph(g_rev)
        print("=== DFS g_rev ====")

    ft_order = topsort(g_rev)

    if debug:
        print("g_rev.len:", g_rev.len)
        print_graph_ext(g_rev)


    if debug:
        print("===== DFS ordered loop =====")


    dfs_ordered_loop(g, ft_order)

    if debug:
        print_graph_ext(g)

    free(ft_order)


""" ################################################################ """
""" ######################### UNIT TESTS ########################### """
""" ################################################################ """


def test_scc_1():
    print_func_name()
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
    print_func_name()
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
    print_func_name()
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
    print_func_name()
    graph = {0: [1],
             1: [0, 2],
             2: [3],
             3: [4],
             4: [2]}
    # graph = {0: [1],
    #          1: [2],
    #          2: [3, 0],
    #          3: [4],
    #          4: [3]}
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

    print(l)
    val, cnt = np.unique(l, return_counts=True)
    print(np.sort(cnt))

    # print(u)
    # print(np.sort(u, axis=1))

def test_scc_big():
    print_func_name()
    graph, graph_rev = read_graphs("scc.txt")

    cdef:
        size_t i
        graph_c * g
        graph_c * g_rev

    g, g_rev = read_graphs("scc.txt")

    print("Running 'scc()' ... ", end="")
    scc(g, g_rev)
    print("done")
    # print_g_ext(g, 100)

    l = np.empty(g.len, dtype=np.uint64)
    cdef size_t [:] l_view = l
    for i in range(g.len):
        l_view[i] = g.node[i].leader

    val, cnt = np.unique(l, return_counts=True)
    print(val[np.argsort(cnt)][-10:])
    print(np.sort(cnt)[-10:])

    free_graph(g)
    free_graph(g_rev)
